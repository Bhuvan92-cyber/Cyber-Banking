"""
audit/signals.py
================
Signal-based audit trail tracking administrative user management actions:
- Connects to User model's pre_save, post_save, and pre_delete signals.
- Captures ROLE_CHANGE, STATUS_TOGGLE, and ACCOUNT_DELETION events.
- Enforces suppression of self-profile updates.
"""
from django.db.models.signals import pre_save, post_save, pre_delete
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import AuditLog


def _get_role_label(is_staff: bool, is_superuser: bool) -> str:
    if is_superuser:
        return "Superadmin"
    if is_staff:
        return "Staff Analyst"
    return "Standard User"


@receiver(pre_save, sender=User)
def capture_user_pre_save_state(sender, instance, **kwargs):
    """
    Captures snapshot of User attributes before database commit
    so changes in is_staff, is_superuser, or is_active can be detected in post_save.
    """
    if instance.pk:
        try:
            old_user = User.objects.get(pk=instance.pk)
            instance._old_state = {
                'id': old_user.id,
                'username': old_user.username,
                'email': old_user.email,
                'is_staff': old_user.is_staff,
                'is_superuser': old_user.is_superuser,
                'is_active': old_user.is_active,
                'role': _get_role_label(old_user.is_staff, old_user.is_superuser),
                'status': 'Active' if old_user.is_active else 'Inactive',
            }
        except User.DoesNotExist:
            instance._old_state = None
    else:
        instance._old_state = None


@receiver(post_save, sender=User)
def log_user_post_save(sender, instance, created, **kwargs):
    """
    Emits AuditLog entries when user permissions, role, or active status change.
    Suppresses logging if the actor is identical to the target user.
    """
    if created:
        return

    old_state = getattr(instance, '_old_state', None)
    if not old_state:
        return

    actor = getattr(instance, '_actor', None)
    ip_address = getattr(instance, '_ip_address', None)
    user_agent = getattr(instance, '_user_agent', '')

    # Requirement: The actor is different from the target (prevent logging self-profile updates)
    if actor and actor.id == instance.id:
        return

    staff_changed = old_state['is_staff'] != instance.is_staff
    superuser_changed = old_state['is_superuser'] != instance.is_superuser
    active_changed = old_state['is_active'] != instance.is_active

    if not (staff_changed or superuser_changed or active_changed):
        return

    # 1. Role or Permission Changes
    if staff_changed or superuser_changed:
        new_role = _get_role_label(instance.is_staff, instance.is_superuser)
        AuditLog.objects.create(
            actor=actor,
            target_user=instance,
            action_type=AuditLog.ActionType.ROLE_CHANGE,
            old_value={
                'username': old_state['username'],
                'is_staff': old_state['is_staff'],
                'is_superuser': old_state['is_superuser'],
                'role': old_state['role'],
            },
            new_value={
                'username': instance.username,
                'is_staff': instance.is_staff,
                'is_superuser': instance.is_superuser,
                'role': new_role,
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )

    # 2. Status Toggles (Active / Inactive)
    if active_changed:
        AuditLog.objects.create(
            actor=actor,
            target_user=instance,
            action_type=AuditLog.ActionType.STATUS_TOGGLE,
            old_value={
                'username': old_state['username'],
                'is_active': old_state['is_active'],
                'status': old_state['status'],
            },
            new_value={
                'username': instance.username,
                'is_active': instance.is_active,
                'status': 'Active' if instance.is_active else 'Inactive',
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )


@receiver(pre_delete, sender=User)
def log_user_pre_delete(sender, instance, **kwargs):
    """
    Emits AuditLog entry when a user is permanently deleted.
    Captures complete pre-deletion state in old_value.
    """
    actor = getattr(instance, '_actor', None)
    ip_address = getattr(instance, '_ip_address', None)
    user_agent = getattr(instance, '_user_agent', '')

    # Requirement: The actor is different from the target
    if actor and actor.id == instance.id:
        return

    old_value = {
        'id': instance.id,
        'username': instance.username,
        'email': instance.email,
        'is_staff': instance.is_staff,
        'is_superuser': instance.is_superuser,
        'is_active': instance.is_active,
        'role': _get_role_label(instance.is_staff, instance.is_superuser),
        'status': 'Active' if instance.is_active else 'Inactive',
    }
    new_value = {
        'deleted': True,
    }

    AuditLog.objects.create(
        actor=actor,
        target_user=instance,
        action_type=AuditLog.ActionType.ACCOUNT_DELETION,
        old_value=old_value,
        new_value=new_value,
        ip_address=ip_address,
        user_agent=user_agent,
    )

