"""
tests/test_admin_user_management.py
===================================
Tests for administrative user management endpoints and RBAC enforcement:
- GET    /api/admin/users/
- PATCH  /api/admin/users/
- PATCH  /api/admin/users/<id>/
- DELETE /api/admin/users/<id>/
"""
import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient

pytestmark = pytest.mark.django_db


@pytest.fixture
def superuser(db):
    return User.objects.create_superuser(
        username="superadmin",
        email="superadmin@bank.internal",
        password="SuperPassword123!",
    )


@pytest.fixture
def superuser_client(api_client, superuser):
    api_client.force_authenticate(user=superuser)
    return api_client


@pytest.fixture
def staff_user(db):
    return User.objects.create_user(
        username="staff_analyst",
        email="analyst@bank.internal",
        password="StaffPassword123!",
        is_staff=True,
    )


@pytest.fixture
def staff_client(api_client, staff_user):
    api_client.force_authenticate(user=staff_user)
    return api_client


@pytest.fixture
def target_user(db):
    return User.objects.create_user(
        username="client_user",
        email="client@fintech.io",
        password="UserPassword123!",
        first_name="Client",
        last_name="One",
    )


class TestAdminUserRBAC:
    url = "/api/admin/users/"

    def test_unauthenticated_access_denied(self, api_client):
        res = api_client.get(self.url)
        assert res.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    def test_standard_user_access_denied(self, auth_client):
        res = auth_client.get(self.url)
        assert res.status_code == status.HTTP_403_FORBIDDEN

    def test_staff_analyst_access_denied_for_admin_endpoint(self, staff_client):
        """Only superusers have permission to manage users."""
        res = staff_client.get(self.url)
        assert res.status_code == status.HTTP_403_FORBIDDEN

    def test_superuser_can_list_all_users(self, superuser_client, target_user, staff_user):
        res = superuser_client.get(self.url)
        assert res.status_code == status.HTTP_200_OK
        usernames = [u["username"] for u in res.data]
        assert "superadmin" in usernames
        assert "client_user" in usernames
        assert "staff_analyst" in usernames

        # Verify computed fields
        client_data = next(u for u in res.data if u["username"] == "client_user")
        assert client_data["role"] == "Standard User"
        assert client_data["status"] == "Active"
        assert client_data["name"] == "Client One"


class TestAdminUserMutations:
    def test_promote_standard_user_to_staff_analyst(self, superuser_client, target_user):
        detail_url = f"/api/admin/users/{target_user.id}/"
        res = superuser_client.patch(detail_url, {"role": "Staff Analyst"}, format="json")
        assert res.status_code == status.HTTP_200_OK
        assert res.data["user"]["role"] == "Staff Analyst"

        target_user.refresh_from_db()
        assert target_user.is_staff is True
        assert target_user.is_superuser is False

    def test_promote_to_superadmin_and_demote_to_standard(self, superuser_client, target_user):
        detail_url = f"/api/admin/users/{target_user.id}/"

        # Promote to Superadmin
        res = superuser_client.patch(detail_url, {"role": "Superadmin"}, format="json")
        assert res.status_code == status.HTTP_200_OK
        target_user.refresh_from_db()
        assert target_user.is_superuser is True
        assert target_user.is_staff is True

        # Downgrade back to Standard User
        res2 = superuser_client.patch(detail_url, {"role": "Standard User"}, format="json")
        assert res2.status_code == status.HTTP_200_OK
        target_user.refresh_from_db()
        assert target_user.is_staff is False
        assert target_user.is_superuser is False

    def test_immediate_rbac_enforcement_after_downgrade(self, superuser_client, staff_user):
        """
        Verify that once Staff Analyst is downgraded to Standard User,
        their /api/auth/me/ response immediately reflects is_staff = False.
        """
        # Step 1: User is currently staff
        user_client = APIClient()
        user_client.force_authenticate(user=staff_user)
        me_res_before = user_client.get("/api/auth/me/")
        assert me_res_before.data["user"]["is_staff"] is True

        # Step 2: Superadmin downgrades user to Standard User
        res = superuser_client.patch(
            f"/api/admin/users/{staff_user.id}/",
            {"role": "Standard User"},
            format="json",
        )
        assert res.status_code == status.HTTP_200_OK

        # Step 3: Refresh user state in DB and verify immediate lack of staff privileges
        staff_user.refresh_from_db()
        assert staff_user.is_staff is False
        assert staff_user.is_superuser is False

        # In DRF test client with force_authenticate, re-binding refreshed instance simulates fresh DB query per HTTP request
        user_client.force_authenticate(user=staff_user)
        me_res_after = user_client.get("/api/auth/me/")
        assert me_res_after.data["user"]["is_staff"] is False

    def test_toggle_user_active_status(self, superuser_client, target_user):
        detail_url = f"/api/admin/users/{target_user.id}/"

        # Deactivate
        res = superuser_client.patch(detail_url, {"is_active": False}, format="json")
        assert res.status_code == status.HTTP_200_OK
        assert res.data["user"]["status"] == "Inactive"
        target_user.refresh_from_db()
        assert target_user.is_active is False

        # Reactivate
        res2 = superuser_client.patch(detail_url, {"is_active": True}, format="json")
        assert res2.status_code == status.HTTP_200_OK
        assert res2.data["user"]["status"] == "Active"
        target_user.refresh_from_db()
        assert target_user.is_active is True

    def test_delete_user_permanently(self, superuser_client, target_user):
        detail_url = f"/api/admin/users/{target_user.id}/"
        res = superuser_client.delete(detail_url)
        assert res.status_code == status.HTTP_200_OK
        assert not User.objects.filter(pk=target_user.id).exists()

    def test_prevent_self_deletion(self, superuser_client, superuser):
        detail_url = f"/api/admin/users/{superuser.id}/"
        res = superuser_client.delete(detail_url)
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot delete your own superuser account" in res.data["error"].lower()
        assert User.objects.filter(pk=superuser.id).exists()

    def test_prevent_self_deactivation(self, superuser_client, superuser):
        detail_url = f"/api/admin/users/{superuser.id}/"
        res = superuser_client.patch(detail_url, {"is_active": False}, format="json")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot deactivate your own superuser account" in res.data["error"].lower()

    def test_prevent_demoting_sole_active_superuser(self, superuser_client, superuser):
        detail_url = f"/api/admin/users/{superuser.id}/"
        res = superuser_client.patch(detail_url, {"role": "Standard User"}, format="json")
        assert res.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot demote the sole active superuser account" in res.data["error"]

