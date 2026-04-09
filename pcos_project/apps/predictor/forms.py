"""Forms for PCOS prediction workflow."""

from __future__ import annotations

from django import forms


YES_NO_CHOICES = [("1", "Yes"), ("0", "No")]
CYCLE_CHOICES = [("regular", "Regular"), ("irregular", "Irregular")]


class PredictionForm(forms.Form):
    """Patient clinical inputs for PCOS risk estimation."""

    age = forms.FloatField(min_value=10, max_value=70, label="Age")
    bmi = forms.FloatField(min_value=10, max_value=70, label="BMI")
    amh_level = forms.FloatField(min_value=0, label="AMH Level (ng/mL)")
    lh_level = forms.FloatField(min_value=0, label="LH Level (mIU/mL)")
    fsh_level = forms.FloatField(min_value=0, label="FSH Level (mIU/mL)")
    follicle_left = forms.IntegerField(min_value=0, label="Follicle Left")
    follicle_right = forms.IntegerField(min_value=0, label="Follicle Right")
    weight_gain = forms.ChoiceField(choices=YES_NO_CHOICES, label="Weight Gain")
    skin_darkening = forms.ChoiceField(choices=YES_NO_CHOICES, label="Skin Darkening")
    hair_growth = forms.ChoiceField(choices=YES_NO_CHOICES, label="Hair Growth")
    pimples = forms.ChoiceField(choices=YES_NO_CHOICES, label="Pimples")
    cycle_regularity = forms.ChoiceField(choices=CYCLE_CHOICES, label="Cycle Regularity")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            css_class = "form-select" if isinstance(field.widget, forms.Select) else "form-control"
            field.widget.attrs.update({"class": css_class})
