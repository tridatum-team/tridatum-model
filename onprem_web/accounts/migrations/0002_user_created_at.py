import time

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('accounts', '0001_initial')]
    operations = [
        # Existing Django accounts have no known creation timestamp.
        migrations.AddField(
            model_name='user', name='created_at', field=models.FloatField(null=True)),
        migrations.AlterField(
            model_name='user', name='created_at', field=models.FloatField(default=time.time, null=True)),
    ]
