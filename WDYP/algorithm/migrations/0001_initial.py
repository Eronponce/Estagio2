# Generated by Django 4.2.4 on 2023-08-29 19:02

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Instance",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("author", models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name="PlantingParameters",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("p", models.FloatField()),
                ("k", models.FloatField()),
                ("temperature", models.FloatField()),
                ("rainfall", models.FloatField()),
                ("humidity", models.FloatField()),
                ("ph", models.FloatField()),
                (
                    "instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="algorithm.instance",
                    ),
                ),
            ],
        ),
    ]
