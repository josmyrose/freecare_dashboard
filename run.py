from app import app
from dashboard import create_dashboard

create_dashboard(app)

if __name__ == '__main__':
    app.run(debug=True)

