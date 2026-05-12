from flask import Flask, render_template, request, jsonify

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from preview_forecast import generate_preview_forecast

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20 per hour"]
)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/forecast-preview", methods=["POST"])
@limiter.limit("10 per hour")

def forecast_preview():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No request data received."
            }), 400

        raw_history = data.get("history", "")

        result = generate_preview_forecast(raw_history)

        return jsonify({
            "status": "success",
            "result": result
        })

    except ValueError as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": "Unexpected server error."
        }), 500


if __name__ == "__main__":
    app.run(debug=False)