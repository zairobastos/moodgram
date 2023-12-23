from analise import modelo, clean_text, converte_string
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
predictor = None


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/submit')
def submit():
    return render_template("submit.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    dados = request.args.get('texto')
    dado = [' '.join(clean_text(str(dados)))]
    previsao = int(predictor.predict(converte_string(dado)))
    return redirect("/resultado/" + str(previsao))


@app.route('/resultado/<int:previsao>')
def resultado(previsao):
    if previsao == 0:
        return render_template("negativo.html")
    elif previsao == 1:
        return render_template("neutro.html")
    elif previsao == 2:
        return render_template("positivo.html")


if __name__ == '__main__':
    predictor, vectorizer = modelo()
    app.run(debug=True)
