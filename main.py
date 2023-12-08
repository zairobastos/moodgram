from modelo import Modelo
from sklearn import svm
from flask import Flask, render_template, request, jsonify,redirect, url_for

app = Flask(__name__)
mod = None
analise = None
predic = None
clf = svm.SVC(kernel='linear', gamma='scale')

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/submit')
def submit():
    return render_template("submit.html")



@app.route('/predict', methods=['POST','GET'])
def predict():
  
  dados = request.args.get('texto')
  dado = [' '.join(mod.clean_text(str(dados)))]
  previsao =int(predic.predict(mod.converte_string(dado)))
  if(previsao == 0):
      return redirect('/negativo')
  elif(previsao == 1):
      return redirect("/neutro")
  elif(previsao == 2):
      return redirect("/positivo")


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
