"""
tests/test_audit_logging.py
===========================
Validates regulatory compliance audit logging:
1. Role change logged correctly with before/after state
2. User deletion logged with full snapshot
3. Self-profile updates suppressed from audit logs
4. IP address and HTTP User Agent accurately recorded
5. Read-only API enforces superuser RBAC and rejects mutations
"""
import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient
from audit.models import AuditLog

pytestmark = pytest.mark.django_db


@pytest.fixture
def superuser(db):
    return User.objects.create_superuser(
        username="audit_superadmin",
        email="audit_admin@bank.internal",
        password="SuperPassword123!",
    )


@pytest.fixture
def superuser_client(superuser):
    client = APIClient()
    client.force_authenticate(user=superuser)
    return client


@pytest.fixture
def staff_user(db):
    return User.objects.create_user(
        username="audit_staff",
        email="staff@bank.internal",
        password="StaffPassword123!",
        is_staff=True,
    )


@pytest.fixture
def target_user(db):
    return User.objects.create_user(
        username="target_client",
        email="client@fintech.io",
        password="TargetPassword123!",
        first_name="Target",
        last_name="User",
    )


def test_1_role_change_logged_correctly(superuser_client, target_user, superuser):
    """Verifies that changing a user's role generates a ROLE_CHANGE audit record."""
    url = f"/api/admin/users/{target_user.id}/"
    res = superuser_client.patch(url, {"role": "Staff Analyst"}, format="json")
    assert res.status_code == status.HTTP_200_OK

    log = AuditLog.objects.filter(target_user=target_user, action_type=AuditLog.ActionType.ROLE_CHANGE).first()
    assert log is not None
    assert log.actor == superuser
    assert log.old_value["role"] == "Standard User"
    assert log.old_value["is_staff"] is False
    assert log.new_value["role"] == "Staff Analyst"
    assert log.new_value["is_staff"] is True


def test_2_deletion_logged_with_old_and_new_values(superuser_client, target_user, superuser):
    """Verifies that permanent account deletion creates an ACCOUNT_DELETION audit record."""
    target_username = target_user.username
    target_id = target_user.id
    url = f"/api/admin/users/{target_id}/"

    res = superuser_client.delete(url)
    assert res.status_code == status.HTTP_200_OK

    log = AuditLog.objects.filter(action_type=AuditLog.ActionType.ACCOUNT_DELETION).first()
    assert log is not None
    assert log.actor == superuser
    assert log.old_value["username"] == target_username
    assert log.old_value["id"] == target_id
    assert log.new_value.get("deleted") is True


def test_3_self_update_not_logged(target_user):
    """Verifies that self-profile mutations are suppressed and NOT recorded in the audit trail."""
    initial_log_count = AuditLog.objects.count()

    # Simulate target user updating their own profile
    target_user._actor = target_user
    target_user.is_staff = True
    target_user.save()

    # Count must remain unchanged because actor == target
    assert AuditLog.objects.count() == initial_log_count


def test_4_ip_and_user_agent_captured(superuser, target_user):
    """Verifies that acting administrator's IP address and User Agent are captured."""
    client = APIClient(REMOTE_ADDR="192.168.1.105", HTTP_USER_AGENT="FintechComplianceAuditor/2.0")
    client.force_authenticate(user=superuser)

    url = f"/api/admin/users/{target_user.id}/"
    res = client.patch(url, {"is_active": False}, format="json")
    assert res.status_code == status.HTTP_200_OK

    log = AuditLog.objects.filter(target_user=target_user, action_type=AuditLog.ActionType.STATUS_TOGGLE).first()
    assert log is not None
    assert log.ip_address == "192.168.1.105"
    assert log.user_agent == "FintechComplianceAuditor/2.0"
    assert log.new_value["status"] == "Inactive"


def test_5_read_only_api_enforces_superuser_permission(superuser, target_user, staff_user):
    """
    Verifies that GET /api/admin/audit-logs/ strictly enforces superuser access,
    resolves actor/target usernames, and rejects mutations with 405 Method Not Allowed.
    """
    # Create sample audit log
    sample_log = AuditLog.objects.create(
        actor=superuser,
        target_user=target_user,
        action_type=AuditLog.ActionType.ROLE_CHANGE,
        old_value={"role": "Standard User"},
        new_value={"role": "Superadmin"},
        ip_address="10.0.0.1",
        user_agent="TestAgent",
    )

    url = "/api/admin/audit-logs/"

    # 1. Unauthenticated -> 401/403
    unauth_client = APIClient()
    assert unauth_client.get(url).status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    # 2. Standard authenticated user -> 403
    std_client = APIClient()
    std_client.force_authenticate(user=target_user)
    assert std_client.get(url).status_code == status.HTTP_403_FORBIDDEN

    # 3. Staff analyst (non-superuser) -> 403
    staff_test_client = APIClient()
    staff_test_client.force_authenticate(user=staff_user)
    assert staff_test_client.get(url).status_code == status.HTTP_403_FORBIDDEN

    # 4. Superuser -> 200 OK with resolved usernames and pagination
    super_client = APIClient()
    super_client.force_authenticate(user=superuser)
    res = super_client.get(url)
    assert res.status_code == status.HTTP_200_OK
    assert "count" in res.data
    assert "results" in res.data
    assert len(res.data["results"]) >= 1

    entry = next(e for e in res.data["results"] if e["id"] == str(sample_log.id))
    assert entry["actor_username"] == superuser.username
    assert entry["target_username"] == target_user.username
    assert entry["action_type"] == "ROLE_CHANGE"

    # 5. Read-only API enforcement: mutating methods must return 405 Method Not Allowed
    assert super_client.post(url, {}).status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    assert super_client.put(url, {}).status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    assert super_client.patch(url, {}).status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    assert super_client.delete(url, {}).status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    # 6. Model immutability enforcement: modifying existing entry directly raises PermissionError
    with pytest.raises(PermissionError, match="immutable and cannot be updated"):
        sample_log.action_type = AuditLog.ActionType.STATUS_TOGGLE
        sample_log.save()

    with pytest.raises(PermissionError, match="immutable and cannot be deleted"):
        sample_log.delete()
