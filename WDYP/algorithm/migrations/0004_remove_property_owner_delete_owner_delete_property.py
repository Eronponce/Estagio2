# Generated by Django 4.2.4 on 2023-10-28 15:12

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('algorithm', '0003_owner_property_delete_formattedvariationknn'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='property',
            name='owner',
        ),
        migrations.DeleteModel(
            name='Owner',
        ),
        migrations.DeleteModel(
            name='Property',
        ),
    ]
