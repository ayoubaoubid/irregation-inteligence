from django import forms


IRRIGATION_NEED_CHOICES = (
    ("Low", "Low"),
    ("Medium", "Medium"),
    ("High", "High"),
)

CROP_GROWTH_STAGE_CHOICES = (
    ("Vegetative", "Vegetative"),
    ("Sowing", "Sowing"),
    ("Flowering", "Flowering"),
    ("Harvest", "Harvest"),
)

IRRIGATION_TYPE_CHOICES = (
    ("Rainfed", "Rainfed"),
    ("Canal", "Canal"),
    ("Drip", "Drip"),
    ("Sprinkler", "Sprinkler"),
)

MULCHING_USED_CHOICES = (
    ("Yes", "Yes"),
    ("No", "No"),
)


class AddDataForm(forms.Form):
    Soil_pH = forms.FloatField(required=True)
    Soil_Moisture = forms.FloatField(required=True)
    Organic_Carbon = forms.FloatField(required=True)
    Electrical_Conductivity = forms.FloatField(required=True)
    Temperature_C = forms.FloatField(required=True)
    Humidity = forms.FloatField(required=True)
    Rainfall_mm = forms.FloatField(required=True)
    Sunlight_Hours = forms.FloatField(required=True)
    Wind_Speed_kmh = forms.FloatField(required=True)
    Crop_Growth_Stage = forms.ChoiceField(choices=CROP_GROWTH_STAGE_CHOICES, required=True)
    Irrigation_Type = forms.ChoiceField(choices=IRRIGATION_TYPE_CHOICES, required=True)
    Field_Area_hectare = forms.FloatField(required=True)
    Mulching_Used = forms.ChoiceField(choices=MULCHING_USED_CHOICES, required=True)
    Previous_Irrigation_mm = forms.FloatField(required=True)
    Irrigation_Need = forms.ChoiceField(choices=IRRIGATION_NEED_CHOICES, required=True)
