# Generated by Django 3.1.1 on 2022-02-28 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0023_evaluationclass_evaluationclassscore'),
    ]

    operations = [
        migrations.AlterField(
            model_name='evaluationclassscore',
            name='evaluation_class_score_id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]