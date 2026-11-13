from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Event',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ts', models.FloatField(db_index=True)),
                ('username', models.CharField(max_length=32, null=True)),
                ('ip', models.CharField(max_length=45, null=True)),
                ('event', models.CharField(max_length=32)),
                ('result', models.CharField(max_length=32)),
                ('detail', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='PurgeGate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('open', models.IntegerField(default=0)),
            ],
        ),
    ]
