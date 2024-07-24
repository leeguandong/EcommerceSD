import os
import argparse
import uvicorn
from ui.web import ecommercesd_ui
from ui.api import app, apply


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manner", default="web", choices=["web", "api"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    if args.manner == "web":
        ecommercesd = ecommercesd_ui()
        ecommercesd.launch(server_name="0.0.0.0", server_port=11245)
    if args.manner == "api":
        this_port = int(os.environ.get('API_PORT', 8381))
        uvicorn.run(app, host="0.0.0.0", port=this_port)
