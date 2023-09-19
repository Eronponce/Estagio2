# Generated by Django 4.2.4 on 2023-09-19 17:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="PlantingInstance",
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
            name="PredictionInfo",
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
                ("model_number", models.IntegerField()),
                ("prediction", models.CharField(max_length=100)),
                ("max_confidence", models.FloatField()),
                ("algorithm_name", models.CharField(max_length=100)),
                ("accuracy", models.FloatField()),
                (
                    "instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="algorithm.plantinginstance",
                    ),
                ),
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
                ("n", models.FloatField(default=0)),
                ("p", models.FloatField(default=0)),
                ("k", models.FloatField(default=0)),
                ("temperature", models.FloatField(default=0)),
                ("humidity", models.FloatField(default=0)),
                ("ph", models.FloatField(default=0)),
                ("rainfall", models.FloatField(default=0)),
                (
                    "instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="algorithm.plantinginstance",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="FormattedVariationKNN",
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
                ("variation", models.CharField(max_length=100)),
                (
                    "instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="algorithm.plantinginstance",
                    ),
                ),
            ],
        ),
    ]