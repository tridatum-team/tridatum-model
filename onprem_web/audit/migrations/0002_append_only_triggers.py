from django.db import migrations

FORWARD = [
    "INSERT INTO audit_purgegate (id, open) VALUES (1, 0)",
    "CREATE TRIGGER audit_event_no_update BEFORE UPDATE ON audit_event "
    "BEGIN SELECT RAISE(ABORT,'append-only'); END",
    "CREATE TRIGGER audit_event_no_delete BEFORE DELETE ON audit_event "
    "WHEN (SELECT open FROM audit_purgegate WHERE id=1)=0 "
    "BEGIN SELECT RAISE(ABORT,'append-only'); END",
]
REVERSE = [
    "DROP TRIGGER audit_event_no_delete",
    "DROP TRIGGER audit_event_no_update",
    "DELETE FROM audit_purgegate WHERE id=1",
]


class Migration(migrations.Migration):
    dependencies = [('audit', '0001_initial')]
    operations = [migrations.RunSQL(sql=FORWARD, reverse_sql=REVERSE)]
