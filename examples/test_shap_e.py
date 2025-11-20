from text_to_3d.shap_e import ShapE

shap_e = ShapE(orientation=[-90.0, 180.0, 0.0])

# shap_e.convert_text_to_3d("a yellow dog", "results/shap_e")
shap_e.convert_multiple_texts_to_3d(["a black dog", "a brown teddy bear", "a white lamp", "a red chair"], "results/shap_e")