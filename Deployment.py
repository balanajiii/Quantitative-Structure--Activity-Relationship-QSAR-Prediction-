from flask import Flask, request, render_template
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/project4.pkl")


def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    canonical_smiles = [Chem.MolToSmiles(mol) for mol in mols if mol is not None]
    return canonical_smiles


def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )
    Mol_descriptors = []
    for mol in mols:
        if mol is not None:
            mol = Chem.AddHs(mol)
            descriptors = calc.CalcDescriptors(mol)
            Mol_descriptors.append(descriptors)
        else:
            Mol_descriptors.append([np.nan] * len(calc.GetDescriptorNames()))
    return np.array(Mol_descriptors)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        smiles = request.form["smiles"]
        canonical = canonical_smiles([smiles])
        descriptors = RDkit_descriptors(canonical)

        # tangani NaN values
        descriptors = np.nan_to_num(descriptors)

        # Debug
        print("Descriptors:", descriptors)

        # prediksi
        prediction = model.predict(descriptors)

        print("Prediction:", prediction)

        return render_template("index.html", prediction=prediction[0], smiles=smiles)
    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
