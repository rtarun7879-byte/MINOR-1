from flask import Flask, render_template, request, redirect, url_for, session, send_file
import hashlib
import sqlite3
import joblib
import numpy as np
import pandas as pd
import os
import csv
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("CROPSENSE_SECRET_KEY", "cropsense2026")
app.config["PERMANENT_SESSION_LIFETIME"] = 86400


# ============================================================
# DEVELOPER SUPER ADMIN
# ============================================================

# For a production deployment, put these in environment variables.
SUPER_ADMIN_USERNAME = os.environ.get(
    "CROPSENSE_SUPER_ADMIN",
    "developer"
)

SUPER_ADMIN_KEY = os.environ.get(
    "CROPSENSE_SUPER_KEY",
    "CS-DEV-2026-SUPER"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ============================================================
# DATABASE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crop.db")

print("Database Path:", DB_PATH)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    # Predictions table.
    # username was added for developer/user statistics.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            crop_name TEXT,
            confidence REAL,
            username TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Existing users table.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Feedback table.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            rating INTEGER DEFAULT 5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Existing crop.db files may have been created before the
    # username column was added. Migrate them without deleting data.
    columns = [
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    ]

    if "username" not in columns:
        conn.execute(
            "ALTER TABLE predictions ADD COLUMN username TEXT"
        )

    conn.commit()
    conn.close()


init_db()


# ============================================================
# CROP EMOJIS
# ============================================================

CROP_EMOJIS = {
    "rice": "🌾",
    "maize": "🌽",
    "chickpea": "🫘",
    "kidneybeans": "🫘",
    "pigeonpeas": "🌿",
    "mothbeans": "🌿",
    "mungbean": "🌿",
    "blackgram": "🌿",
    "lentil": "🫘",
    "pomegranate": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "muskmelon": "🍈",
    "apple": "🍎",
    "orange": "🍊",
    "papaya": "🍈",
    "coconut": "🥥",
    "cotton": "🌸",
    "jute": "🌿",
    "coffee": "☕"
}


# ============================================================
# AUTH ROUTES
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not password:
            return render_template(
                "signup.html",
                error="Username and password are required."
            )

        if password != confirm:
            return render_template(
                "signup.html",
                error="Passwords do not match."
            )

        hashed = hashlib.sha256(
            password.encode("utf-8")
        ).hexdigest()

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            return render_template(
                "signup.html",
                error="Username already exists."
            )

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        hashed = hashlib.sha256(
            password.encode("utf-8")
        ).hexdigest()

        conn = get_db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            AND password = ?
            """,
            (username, hashed)
        ).fetchone()

        conn.close()

        if user:
            session.permanent = True
            session["user"] = username
            return redirect(url_for("home"))

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ============================================================
# MAIN ROUTES
# ============================================================

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions"
    ).fetchone()["c"]

    latest = conn.execute(
        """
        SELECT crop_name, confidence, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 3
        """
    ).fetchall()

    conn.close()

    return render_template(
        "index.html",
        total=total,
        latest=latest,
        emojis=CROP_EMOJIS
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            nitrogen = float(request.form["nitrogen"])
            phosphorus = float(request.form["phosphorus"])
            potassium = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

        except (ValueError, KeyError):
            return render_template(
                "predict.html",
                error="Please fill all fields with valid numbers."
            )

        # Server-side range validation.
        if not (0 <= nitrogen <= 200):
            return render_template(
                "predict.html",
                error="Nitrogen must be between 0 and 200 kg/ha."
            )

        if not (0 <= phosphorus <= 200):
            return render_template(
                "predict.html",
                error="Phosphorus must be between 0 and 200 kg/ha."
            )

        if not (0 <= potassium <= 200):
            return render_template(
                "predict.html",
                error="Potassium must be between 0 and 200 kg/ha."
            )

        if not (-10 <= temperature <= 60):
            return render_template(
                "predict.html",
                error="Temperature must be between -10 and 60 °C."
            )

        if not (0 <= humidity <= 100):
            return render_template(
                "predict.html",
                error="Humidity must be between 0 and 100%."
            )

        if not (0 <= rainfall <= 3000):
            return render_template(
                "predict.html",
                error="Rainfall must be between 0 and 3000 mm."
            )

        if not (0 <= ph <= 14):
            return render_template(
                "predict.html",
                error="Soil pH must be between 0 and 14."
            )

        features = pd.DataFrame(
            [[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )

        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]

        # prediction[0] is the encoded class index.
        class_index = prediction[0]

        confidence = round(
            float(probabilities[class_index]) * 100,
            2
        )

        crop_name = label_encoder.inverse_transform(
            prediction
        )[0]

        conn = get_db()

        cursor = conn.execute(
            """
            INSERT INTO predictions (
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
                crop_name,
                confidence,
                username
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
                crop_name,
                confidence,
                session["user"]
            )
        )

        record_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return redirect(
            url_for("result", id=record_id)
        )

    return render_template("predict.html")


@app.route("/result/<int:id>")
def result(id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    record = conn.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (id,)
    ).fetchone()

    conn.close()

    if not record:
        return redirect(url_for("predict"))

    # A user can view a record if it exists. The existing app
    # behaved this way; no ownership restriction is introduced.
    emoji = CROP_EMOJIS.get(
        record["crop_name"],
        "🌱"
    )

    return render_template(
        "result.html",
        record=record,
        emoji=emoji
    )


# ============================================================
# FEEDBACK
# ============================================================

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        message = request.form.get(
            "message",
            ""
        ).strip()

        rating = request.form.get(
            "rating",
            "5"
        )

        if not message:
            return render_template(
                "feedback.html",
                error="Please enter your feedback."
            )

        try:
            rating = int(rating)
        except (TypeError, ValueError):
            rating = 5

        rating = max(1, min(5, rating))

        conn = get_db()

        conn.execute(
            """
            INSERT INTO feedback (
                username,
                message,
                rating
            )
            VALUES (?, ?, ?)
            """,
            (
                session["user"],
                message,
                rating
            )
        )

        conn.commit()
        conn.close()

        return redirect(url_for("home"))

    return render_template("feedback.html")


# ============================================================
# DEVELOPER SUPER ADMIN
# ============================================================

@app.route(
    "/super-admin",
    methods=["GET", "POST"]
)
def super_admin_login():
    if session.get("super_admin"):
        return redirect(
            url_for("super_admin_dashboard")
        )

    if request.method == "POST":
        username = request.form.get(
            "username",
            ""
        ).strip()

        super_key = request.form.get(
            "super_key",
            ""
        ).strip()

        if (
            username == SUPER_ADMIN_USERNAME
            and super_key == SUPER_ADMIN_KEY
        ):
            session.permanent = True
            session["super_admin"] = True

            return redirect(
                url_for("super_admin_dashboard")
            )

        return render_template(
            "super_admin_login.html",
            error="Invalid developer credentials."
        )

    return render_template(
        "super_admin_login.html"
    )


@app.route("/super-admin/dashboard")
def super_admin_dashboard():
    if not session.get("super_admin"):
        return redirect(
            url_for("super_admin_login")
        )

    conn = get_db()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    total_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    total_feedback = conn.execute(
        "SELECT COUNT(*) FROM feedback"
    ).fetchone()[0]

    feedback_rows = conn.execute(
        """
        SELECT *
        FROM feedback
        ORDER BY id DESC
        """
    ).fetchall()

    users = conn.execute(
        """
        SELECT
            u.username,
            COUNT(p.id) AS prediction_count
        FROM users AS u
        LEFT JOIN predictions AS p
            ON u.username = p.username
        GROUP BY u.username
        ORDER BY prediction_count DESC
        """
    ).fetchall()

    recent_predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT 20
        """
    ).fetchall()

    # Crop statistics.
    crop_stats = conn.execute(
        """
        SELECT
            crop_name,
            COUNT(*) AS prediction_count,
            ROUND(AVG(confidence), 2) AS average_confidence
        FROM predictions
        GROUP BY crop_name
        ORDER BY prediction_count DESC
        """
    ).fetchall()

    # Average feedback rating.
    average_rating = conn.execute(
        """
        SELECT ROUND(AVG(rating), 2)
        FROM feedback
        """
    ).fetchone()[0]

    if average_rating is None:
        average_rating = 0

    conn.close()

    return render_template(
        "super_admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        total_feedback=total_feedback,
        average_rating=average_rating,
        feedback=feedback_rows,
        users=users,
        predictions=recent_predictions,
        crop_stats=crop_stats
    )


@app.route("/super-admin/logout")
def super_admin_logout():
    session.pop(
        "super_admin",
        None
    )

    return redirect(
        url_for("super_admin_login")
    )


# ============================================================
# RECORDS
# ============================================================

@app.route("/records")
def records():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    crops = conn.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "records.html",
        crops=crops,
        emojis=CROP_EMOJIS
    )


@app.route(
    "/records/delete/<int:id>",
    methods=["POST"]
)
def delete_record(id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    conn.execute(
        "DELETE FROM predictions WHERE id = ?",
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for("records"))


@app.route(
    "/records/clear",
    methods=["POST"]
)
def clear_records():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    conn.execute(
        "DELETE FROM predictions"
    )

    # Reset SQLite AUTOINCREMENT counter if present.
    conn.execute(
        "DELETE FROM sqlite_sequence "
        "WHERE name = 'predictions'"
    )

    conn.commit()
    conn.close()

    return redirect(url_for("records"))


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("about.html")


# ============================================================
# PDF DOWNLOAD
# ============================================================

@app.route("/download/pdf/<int:id>")
def download_pdf(id):
    conn = get_db()

    record = conn.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (id,)
    ).fetchone()

    conn.close()

    if not record:
        return redirect(url_for("records"))

    pdf_file = os.path.join(
        app.root_path,
        f"prediction_{id}.pdf"
    )

    c = canvas.Canvas(
        pdf_file,
        pagesize=A4
    )

    width, height = A4

    # Header.
    c.setFillColor(
        colors.HexColor("#0d4a1e")
    )

    c.rect(
        0,
        height - 90,
        width,
        90,
        fill=1,
        stroke=0
    )

    c.setFillColor(colors.white)

    c.setFont(
        "Helvetica-Bold",
        24
    )

    c.drawString(
        45,
        height - 52,
        "CropSense"
    )

    c.setFont(
        "Helvetica",
        11
    )

    c.drawString(
        45,
        height - 70,
        "AI Crop Recommendation Report"
    )

    # Crop image.
    image_path = os.path.join(
        BASE_DIR,
        "static",
        "images",
        f"{record['crop_name']}.jpg"
    )

    image_y = height - 330

    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((260, 210))

            c.drawImage(
                ImageReader(img),
                45,
                image_y,
                width=260,
                height=210,
                preserveAspectRatio=True,
                anchor="sw",
                mask="auto"
            )

        except Exception as exc:
            print(
                "PDF image error:",
                exc
            )

    # Recommendation card.
    c.setFillColor(
        colors.HexColor("#e6f9ec")
    )

    c.roundRect(
        330,
        image_y,
        220,
        210,
        16,
        fill=1,
        stroke=0
    )

    c.setFillColor(
        colors.HexColor("#0d4a1e")
    )

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        350,
        image_y + 175,
        "RECOMMENDED CROP"
    )

    c.setFont(
        "Helvetica-Bold",
        24
    )

    c.setFillColor(
        colors.HexColor("#1a7a35")
    )

    c.drawString(
        350,
        image_y + 135,
        record["crop_name"].title()
    )

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.setFillColor(
        colors.HexColor("#3d5a40")
    )

    c.drawString(
        350,
        image_y + 95,
        "Confidence"
    )

    c.setFont(
        "Helvetica-Bold",
        20
    )

    c.setFillColor(
        colors.HexColor("#00a889")
    )

    c.drawString(
        350,
        image_y + 65,
        f"{record['confidence']}%"
    )

    # Input summary.
    summary_top = image_y - 35

    c.setFillColor(
        colors.HexColor("#0d4a1e")
    )

    c.setFont(
        "Helvetica-Bold",
        15
    )

    c.drawString(
        45,
        summary_top,
        "Input Parameters"
    )

    rows = [
        (
            "Nitrogen (N)",
            f"{record['nitrogen']} kg/ha"
        ),
        (
            "Phosphorus (P)",
            f"{record['phosphorus']} kg/ha"
        ),
        (
            "Potassium (K)",
            f"{record['potassium']} kg/ha"
        ),
        (
            "Temperature",
            f"{record['temperature']} °C"
        ),
        (
            "Humidity",
            f"{record['humidity']} %"
        ),
        (
            "Soil pH",
            f"{record['ph']}"
        ),
        (
            "Rainfall",
            f"{record['rainfall']} mm"
        ),
        (
            "Prediction",
            record["crop_name"].title()
        ),
        (
            "Generated",
            str(record["created_at"])[:19]
            if record["created_at"]
            else "—"
        )
    ]

    y = summary_top - 28

    for label, value in rows:

        c.setFillColor(
            colors.HexColor("#f3f8f4")
        )

        c.roundRect(
            45,
            y - 5,
            505,
            23,
            5,
            fill=1,
            stroke=0
        )

        c.setFillColor(
            colors.HexColor("#3d5a40")
        )

        c.setFont(
            "Helvetica-Bold",
            9.5
        )

        c.drawString(
            57,
            y + 2,
            label
        )

        c.setFillColor(
            colors.HexColor("#0f1f0f")
        )

        c.setFont(
            "Helvetica",
            9.5
        )

        c.drawRightString(
            535,
            y + 2,
            value
        )

        y -= 29

    # Footer.
    c.setStrokeColor(
        colors.HexColor("#d8e8dc")
    )

    c.line(
        45,
        38,
        width - 45,
        38
    )

    c.setFillColor(
        colors.HexColor("#7a9e7e")
    )

    c.setFont(
        "Helvetica",
        8
    )

    c.drawString(
        45,
        23,
        "CropSense • Powered by Random Forest ML • Built with Flask"
    )

    c.drawRightString(
        width - 45,
        23,
        "Prediction Report"
    )

    c.save()

    return send_file(
        pdf_file,
        as_attachment=True,
        download_name=f"CropSense_Prediction_{id}.pdf"
    )


# ============================================================
# SINGLE CSV DOWNLOAD
# ============================================================

@app.route("/download/csv/<int:id>")
def download_csv(id):
    conn = get_db()

    record = conn.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (id,)
    ).fetchone()

    conn.close()

    if not record:
        return redirect(url_for("records"))

    csv_file = os.path.join(
        app.root_path,
        f"prediction_{id}.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Crop",
            "Confidence",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "pH",
            "Rainfall",
            "Username"
        ])

        writer.writerow([
            record["crop_name"],
            record["confidence"],
            record["nitrogen"],
            record["phosphorus"],
            record["potassium"],
            record["temperature"],
            record["humidity"],
            record["ph"],
            record["rainfall"],
            record["username"] or ""
        ])

    return send_file(
        csv_file,
        as_attachment=True,
        download_name=f"CropSense_Prediction_{id}.csv"
    )


# ============================================================
# PNG DOWNLOAD
# ============================================================

@app.route("/download/png/<int:id>")
def download_png(id):
    conn = get_db()

    record = conn.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (id,)
    ).fetchone()

    conn.close()

    if not record:
        return redirect(url_for("records"))

    # Complete PNG report.
    W, H = 1200, 1450

    bg = (8, 31, 13)
    card = (18, 55, 30)
    light = (230, 249, 236)
    white = (255, 255, 255)
    muted = (190, 215, 195)
    green = (77, 204, 110)
    teal = (0, 201, 167)

    report = Image.new(
        "RGB",
        (W, H),
        bg
    )

    draw = ImageDraw.Draw(report)

    def load_font(size, bold=False):
        candidates = [
            (
                "C:/Windows/Fonts/arialbd.ttf"
                if bold
                else "C:/Windows/Fonts/arial.ttf"
            ),
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            )
        ]

        for path in candidates:
            if os.path.exists(path):
                return ImageFont.truetype(
                    path,
                    size
                )

        return ImageFont.load_default()

    f_title = load_font(48, True)
    f_sub = load_font(23)
    f_crop = load_font(42, True)
    f_section = load_font(27, True)
    f_label = load_font(22, True)
    f_value = load_font(22)
    f_small = load_font(18)

    # Header.
    draw.rounded_rectangle(
        (35, 30, W - 35, 170),
        radius=28,
        fill=(13, 74, 30)
    )

    draw.text(
        (70, 58),
        "CropSense",
        font=f_title,
        fill=white
    )

    draw.text(
        (72, 115),
        "AI Crop Recommendation Report",
        font=f_sub,
        fill=muted
    )

    # Crop image card.
    image_box = (
        55,
        205,
        565,
        625
    )

    draw.rounded_rectangle(
        image_box,
        radius=28,
        fill=card
    )

    image_path = os.path.join(
        BASE_DIR,
        "static",
        "images",
        f"{record['crop_name']}.jpg"
    )

    if os.path.exists(image_path):
        try:
            crop_img = Image.open(
                image_path
            ).convert("RGB")

            crop_img.thumbnail(
                (465, 360)
            )

            x = (
                image_box[0]
                + (
                    image_box[2]
                    - image_box[0]
                    - crop_img.width
                ) // 2
            )

            y_img = (
                image_box[1]
                + (
                    image_box[3]
                    - image_box[1]
                    - crop_img.height
                ) // 2
            )

            report.paste(
                crop_img,
                (x, y_img)
            )

        except Exception as exc:
            print(
                "PNG image error:",
                exc
            )

    # Recommendation card.
    draw.rounded_rectangle(
        (610, 205, 1145, 625),
        radius=28,
        fill=light
    )

    draw.text(
        (650, 245),
        "RECOMMENDED CROP",
        font=f_section,
        fill=(13, 74, 30)
    )

    draw.text(
        (650, 310),
        record["crop_name"].title(),
        font=f_crop,
        fill=(26, 122, 53)
    )

    draw.text(
        (650, 405),
        "Confidence",
        font=f_label,
        fill=(61, 90, 64)
    )

    draw.text(
        (650, 445),
        f"{record['confidence']}%",
        font=f_crop,
        fill=teal
    )

    # Input summary.
    draw.text(
        (55, 685),
        "Input Parameters",
        font=f_section,
        fill=green
    )

    rows = [
        (
            "Nitrogen (N)",
            f"{record['nitrogen']} kg/ha"
        ),
        (
            "Phosphorus (P)",
            f"{record['phosphorus']} kg/ha"
        ),
        (
            "Potassium (K)",
            f"{record['potassium']} kg/ha"
        ),
        (
            "Temperature",
            f"{record['temperature']} °C"
        ),
        (
            "Humidity",
            f"{record['humidity']} %"
        ),
        (
            "Soil pH",
            f"{record['ph']}"
        ),
        (
            "Rainfall",
            f"{record['rainfall']} mm"
        ),
        (
            "Prediction",
            record["crop_name"].title()
        ),
        (
            "Generated",
            str(record["created_at"])[:19]
            if record["created_at"]
            else "—"
        )
    ]

    y = 735

    for label, value in rows:

        draw.rounded_rectangle(
            (55, y, 1145, y + 54),
            radius=12,
            fill=card
        )

        draw.text(
            (80, y + 13),
            label,
            font=f_label,
            fill=white
        )

        bbox = draw.textbbox(
            (0, 0),
            value,
            font=f_value
        )

        value_w = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (1115 - value_w, y + 14),
            value,
            font=f_value,
            fill=muted
        )

        y += 70

    # Footer.
    draw.line(
        (55, 1375, 1145, 1375),
        fill=(65, 105, 75),
        width=2
    )

    draw.text(
        (55, 1390),
        "CropSense • Powered by Random Forest ML • Built with Flask",
        font=f_small,
        fill=muted
    )

    png_file = os.path.join(
        app.root_path,
        f"prediction_{id}_report.png"
    )

    report.save(
        png_file,
        "PNG",
        optimize=True
    )

    return send_file(
        png_file,
        as_attachment=True,
        download_name=f"CropSense_Prediction_{id}.png",
        mimetype="image/png"
    )


# ============================================================
# DOWNLOAD ALL CSV
# ============================================================

@app.route("/download_all_csv")
def download_all_csv():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    records = conn.execute(
        "SELECT * FROM predictions ORDER BY id"
    ).fetchall()

    conn.close()

    csv_file = os.path.join(
        app.root_path,
        "all_predictions.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "ID",
            "Crop",
            "Confidence",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "pH",
            "Rainfall",
            "Username",
            "Date"
        ])

        for row in records:

            writer.writerow([
                row["id"],
                row["crop_name"],
                row["confidence"],
                row["nitrogen"],
                row["phosphorus"],
                row["potassium"],
                row["temperature"],
                row["humidity"],
                row["ph"],
                row["rainfall"],
                row["username"] or "",
                row["created_at"]
            ])

    return send_file(
        csv_file,
        as_attachment=True,
        download_name="CropSense_All_Predictions.csv",
        mimetype="text/csv"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )