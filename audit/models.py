"""
audit/models.py
===============
Immutable audit trail for banking regulatory compliance.
Tracks all administrative user management and permission changes.
"""
import uuid
from django.db import models
from django.contrib.auth.models import User


class AuditLog(models.Model):
    class ActionType(models.TextChoices):
        ROLE_CHANGE = 'ROLE_CHANGE', 'Role Change'
        STATUS_TOGGLE = 'STATUS_TOGGLE', 'Status Toggle'
        ACCOUNT_DELETION = 'ACCOUNT_DELETION', 'Account Deletion'
        PERMISSION_UPDATE = 'PERMISSION_UPDATE', 'Permission Update'

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique immutable UUID identifier"
    )
    actor = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_actions_performed',
        help_text="The admin who performed the action"
    )
    target_user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audit_records',
        help_text="The user who was modified"
    )
    action_type = models.CharField(
        max_length=50,
        choices=ActionType.choices,
        help_text="Type of action performed: ROLE_CHANGE, STATUS_TOGGLE, ACCOUNT_DELETION, PERMISSION_UPDATE"
    )
    old_value = models.JSONField(
        default=dict,
        blank=True,
        help_text="State of user prior to action"
    )
    new_value = models.JSONField(
        default=dict,
        blank=True,
        help_text="State of user after action"
    )
    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="Client IP address of the acting admin"
    )
    user_agent = models.TextField(
        max_length=500,
        blank=True,
        default='',
        help_text="HTTP User Agent string of the acting admin"
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        editable=False,
        help_text="UTC timestamp when action occurred"
    )

    class Meta:
        db_table = 'audit_logs'
        ordering = ['-timestamp']
        verbose_name = 'Audit Log'
        verbose_name_plural = 'Audit Logs'

    def save(self, *args, **kwargs):
        # Enforce append-only immutability
        if self.pk and not self._state.adding:
            # Check if this row already exists in the database
            if AuditLog.objects.filter(pk=self.pk).exists():
                raise PermissionError("Audit logs are immutable and cannot be updated.")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise PermissionError("Audit logs are immutable and cannot be deleted.")

    def __str__(self):
        actor_name = self.actor.username if self.actor else "System"
        target_name = (
            self.target_user.username
            if self.target_user
            else (self.old_value.get('username') or "Deleted User")
        )
        return f"[{self.timestamp}] {self.action_type} by {actor_name} on {target_name}"

