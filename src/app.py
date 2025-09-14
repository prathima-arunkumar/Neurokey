# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import hashlib
from eeg_module import enroll_user, authenticate_user, generate_key, store_secret, retrieve_secret
import psycopg2
import numpy as np

app = Flask(__name__)
app.secret_key = 'bmscollege'


DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'root',
    'port': 5432
}


users = {}
secrets = {}


def insert_neurokey(user_id, task, neurokey):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO neurokeys (user_id, task, neurokey)
            VALUES (%s, %s, %s)
        """, (user_id, task, neurokey))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print("❌ Error inserting into DB:", e)
        return False

@app.route('/')
def home():
    return redirect(url_for('enroll'))

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    valid_tasks = {'reading', 'writing', 'imagination'}
    if request.method == 'POST':
        user_id = request.form['user_id']
        task = request.form['task'].strip().lower()

        if task not in valid_tasks:
            flash("❌ Invalid task. Please enter 'reading', 'writing', or 'imagination'.", 'danger')
            return render_template('enroll.html')

        success = enroll_user(user_id, task)
        if success:
            flash(f"✅ Enrollment complete for '{user_id}' task: '{task}'", 'success')
        else:
            flash("❌ Enrollment failed.", 'danger')
    return render_template('enroll.html')


@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    result = None
    if request.method == 'POST':
        user_id = request.form['user_id']
        task = request.form['task']
        success, features, predicted_task, confidence = authenticate_user(user_id, task)
        result = {
            'success': success,
            'predicted': predicted_task,
            'confidence': confidence,
            'user_id': user_id,
            'task': task
        }
        return render_template("authenticate.html", result=result)


    return render_template("authenticate.html")

# @app.route('/keygen')
# def keygen():
#     user_id = request.args.get('user_id')
#     task = request.args.get('task')
#     key = generate_key(user_id, task)
#     return render_template('keygen.html', key=key, user_id=user_id)


@app.route('/generate_key', methods=['POST', 'GET'])
def generate_key_route():
    if request.method == 'POST':
        user_id = request.form['user_id']
        task = request.form['task']

        selected, binary, key_hash = generate_key(user_id, task)

        # 🔐 Ensure key_hash is a plain string
        if isinstance(key_hash, np.ndarray):
            key_hash = str(key_hash.item())
        else:
            key_hash = str(key_hash)

        inserted = insert_neurokey(user_id, task, key_hash)
        if inserted:
            flash("✅ Key saved to database.", "success")
        else:
            flash("⚠️ Key generated but not saved to database.", "warning")

        if key_hash:
            return render_template("keygen.html",
                                   user_id=user_id,
                                   task=task,
                                   vectorized=selected.tolist(),
                                   binary=binary.tolist(),
                                   key_hash=key_hash)
        else:
            return render_template("keygen.html", error="❌ Key generation failed.")

    return redirect(url_for('authenticate'))






@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        task = request.form['task']

        # Authenticate
        success, features, predicted_task, confidence = authenticate_user(user_id, task)
        if success:
            key_hash = generate_key(user_id, task)

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("SELECT neurokey FROM neurokeys WHERE user_id = %s", (user_id,))
            record = cur.fetchone()
            cur.close()
            conn.close()

            if record or record[0] == key_hash:
                flash("✅ Login Successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("❌ Key mismatch. Try using 'Forgot Task' option.", "danger")
        else:
            flash("❌ Authentication failed.", "danger")

    return render_template('login.html')

@app.route('/regenerate_key', methods=['GET', 'POST'])
def regenerate_key():
    if request.method == 'POST':
        user_id = request.form['user_id']
        new_task = request.form['new_task']

        # Try to authenticate using the new task
        success, features, predicted_task, confidence = authenticate_user(user_id, new_task)

        if success:
            # Generate new key hash from authenticated features
            key_hash = generate_key(user_id, new_task)

            if isinstance(key_hash, np.ndarray):
                key_hash = str(key_hash.item())
            else:
                key_hash = str(key_hash)

            # Update only the key_hash in database
            conn = psycopg2.connect(
                dbname="neurokey",
                user="postgres",
                password="Joel@123",
                host="localhost",
                port="5432"
            )
            cur = conn.cursor()

            cur.execute("""
                UPDATE neurokeys
                SET neurokey = %s,
                    task = %s
                WHERE user_id = %s;
            """, (key_hash, new_task, user_id))

            conn.commit()
            cur.close()
            conn.close()

            flash(f"✅ Key regenerated successfully using new task '{new_task}'.", "success")
            return redirect(url_for('login'))
        else:
            flash("❌ Re-authentication with new task failed. Cannot regenerate key.", "danger")

    return render_template("regenerate_key.html")



# @app.route('/vault', methods=['GET', 'POST'])
# def vault():
#     key = request.args.get('key')
#     secret = None
#     if request.method == 'POST':
#         new_secret = request.form['secret']
#         store_secret(key, new_secret)
#         flash("✅ Secret stored securely!", 'success')
#     secret = retrieve_secret(key)
#     return render_template('vault.html', secret=secret)


# @app.route('/store_secret', methods=['GET', 'POST'])
# def store_secret_route():
#     if request.method == 'POST':
#         key = request.form['key']
#         secret = request.form['secret']
#         success = store_secret(key, secret)
#         if success:
#             return render_template("store_secret.html", message="✅ Secret stored securely.")
#         else:
#             return render_template("store_secret.html", error="❌ Failed to store secret.")
#     return render_template("store_secret.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")  # Any protected content


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    app.run(debug=True)
