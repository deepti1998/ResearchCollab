# Generated by Django 2.2.7 on 2019-11-30 08:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('searchQuery', '0003_auto_20191130_1416'),
    ]

    operations = [
        migrations.AlterField(
            model_name='documentdetails',
            name='abstract',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='documentdetails',
            name='authors',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='documentdetails',
            name='references',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='documentdetails',
            name='title',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='documentdetails',
            name='venue',
            field=models.TextField(null=True),
        ),
    ]
