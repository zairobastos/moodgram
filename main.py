from modelo import Modelo
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
mod = None
analise = None
predic = None

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/submit')
def submit():
    return render_template("submit.html")



@app.route('/predict', methods=['POST'])
def predict():
  dados = request.get_json(force=True)
  print(dados["texto"])
  print(predic)
  dado = [' '.join(mod.clean_text(dados["texto"]))]
  previsao =predic.predict(mod.converte_string(dado))
  resposta = {'Sentimento' :previsao[0]}
  print(resposta)
  return jsonify(resposta)


@app.route('/positivo')
def positivo():
    return render_template("positivo.html")


@app.route('/neutro')
def neutro():
    return render_template("neutro.html")


@app.route('/negativo')
def negativo():
    return render_template("negativo.html")


if __name__ == '__main__':
    mod = Modelo()
    analise = mod.analise()
    predic = mod.SVM(analise)
    app.run(debug=True)
