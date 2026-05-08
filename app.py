import os
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

from recommender import RecommendationEngine
from try_on import run_tryon_pipeline
from size_estimation import BodyMeasurementSystem

app = Flask(__name__)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("csv1.csv")
engine = RecommendationEngine("csv1.csv")

# -----------------------------
# IMAGE FOLDER
# -----------------------------
IMAGE_FOLDER = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\try_on_img"

@app.route('/product_image/<filename>')
def product_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


# -----------------------------
# HELPER: CLEAN INPUT
# -----------------------------
def clean(val):
    if val is None:
        return None
    val = val.strip()
    return val if val != "" else None


# -----------------------------
# HOME (RECOMMEND + TRYON UI)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # -----------------------------
        # TRY-ON BUTTON CLICK
        # -----------------------------
        if request.form.get("action") == "tryon":

            product_id = int(request.form["product_id"])
            row = df[df["product_id"] == product_id].iloc[0]

            return render_template(
                "results.html",
                show_tryon=True,
                selected_product=row.to_dict(),
                results=None
            )

        # -----------------------------
        # NORMAL RECOMMENDATION
        # -----------------------------
        community = clean(request.form.get("community"))
        category = clean(request.form.get("category"))

        min_price = request.form.get("min_price")
        max_price = request.form.get("max_price")

        min_price = int(min_price) if min_price and min_price.strip() != "" else None
        max_price = int(max_price) if max_price and max_price.strip() != "" else None

        user_profile = {
            "community": community,
            "category": category,
            "price_range": (min_price, max_price)
        }

        # 🔥 DEBUG (remove later)
        print("USER PROFILE:", user_profile)

        results = engine.recommend(user_profile).to_dict("records")

        return render_template(
            "results.html",
            results=results,
            show_tryon=False
        )

    return render_template("index.html")


# -----------------------------
# RUN TRY-ON PIPELINE
# -----------------------------
@app.route("/run_tryon", methods=["POST"])
def run_tryon():

    product_id = int(request.form["product_id"])
    file = request.files["person_image"]

    # save uploaded image
    os.makedirs("uploads", exist_ok=True)
    person_path = os.path.join("uploads", file.filename)
    file.save(person_path)

    # get product
    row = df[df["product_id"] == product_id].iloc[0]
    selected_product = row.to_dict()

    # run pipeline
    output_path = "final_result.png"

    run_tryon_pipeline(
        person_path_input=person_path,
        product_id_input=product_id,
        output_path_input=output_path
    )

    return render_template(
        "results.html",
        show_tryon=True,
        selected_product=selected_product,
        result_image=output_path,
        last_uploaded_image=person_path,
        results=None
    )


# -----------------------------
# SIZE RECOMMENDATION
# -----------------------------
@app.route("/get_size", methods=["POST"])
def get_size():

    product_id = int(request.form["product_id"])
    height = float(request.form["height"])
    image_path = request.form["image_path"]

    row = df[df["product_id"] == product_id].iloc[0]
    selected_product = row.to_dict()

    # -----------------------------
    # RULE 1: FREE SIZE
    # -----------------------------
    if str(row["size"]).strip().lower() == "free_size":
        size_result = "Free Size"

    else:
        # -----------------------------
        # RULE 2: ML MODEL
        # -----------------------------
        gender = row["gender"]

        system = BodyMeasurementSystem()
        res = system.get_measurements_and_size(
            image_path,
            height,
            gender
        )

        size_result = res.get("recommended_size", "Error")

    return render_template(
        "results.html",
        show_tryon=True,
        selected_product=selected_product,
        size_result=size_result,
        last_uploaded_image=image_path,
        results=None
    )


# -----------------------------
# SERVE RESULT IMAGE
# -----------------------------
@app.route('/result_image/<filename>')
def result_image(filename):
    return send_from_directory(".", filename)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
