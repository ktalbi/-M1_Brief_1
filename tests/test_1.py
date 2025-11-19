from models.models import create_nn_model, train_model as train_model_impl, model_predict
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from modules.preprocess import preprocessing, split


def generate_sample_data():
    # Nombre de lignes à générer
    n = 50

    # Valeurs possibles pour chaque colonne
    prenoms = ["Zoé", "Emma", "Lucas", "Hugo", "Léo", "Mila", "Noah", "Jules", "Lina", "Victor"]
    noms = ["Martin", "Dupont", "Bernard", "Robert", "Richard", "Durand", "Moreau", "Simon", "Laurent", "Petit"]
    sexes = ["M", "F"]
    sport = ["oui", "non"]
    etudes = ["aucun", "bac", "bac+1", "bac+2", "bac+3", "master", "doctorat"]
    regions = ["Normandie", "Île-de-France", "Bretagne", "Occitanie", "Grand Est", "Nouvelle-Aquitaine"]
    smoker_choice = ["oui", "non"]
    fr_choice = ["oui", "non"]

    # Génération des lignes aléatoires
    df_random = pd.DataFrame({
        "nom": np.random.choice(noms, n),
        "prenom": np.random.choice(prenoms, n),
        "age": np.random.randint(18, 90, n),
        "taille": np.round(np.random.uniform(150, 200, n), 1),
        "poids": np.round(np.random.uniform(45, 110, n), 1),
        "sexe": np.random.choice(sexes, n),
        "sport_licence": np.random.choice(sport, n),
        "niveau_etude": np.random.choice(etudes, n),
        "region": np.random.choice(regions, n),
        "smoker": np.random.choice(smoker_choice, n),
        "nationalité_francaise": np.random.choice(fr_choice, n),
        "revenu_estime_mois": np.random.randint(900, 6000, n),
        "montant_pret": np.round(np.random.uniform(500, 50000, n), 2)
    })

    X, y, _ = preprocessing(df_random)
    X_train, X_test, y_train, y_test = split(X, y)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        # pour les tests on utilise X_test comme validation
        "X_val": X_test,
        "y_val": y_test,
        "epochs": 2,
        "batch_size": 4,
    }

    print('**** data *****')
    return data


def train_model_for_test():
    """Entraîne un modèle avec des données synthétiques pour les tests."""
    data = generate_sample_data()

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    epochs = data['epochs']
    batch_size = data['batch_size']

    model = create_nn_model(X_train.shape[1])

    # Appel de la vraie fonction train_model importée (renommée train_model_impl)
    model, hist = train_model_impl(
        model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, hist, data


def test_create_nn_model_has_correct_input_dim_and_predict():
    input_dim = 5
    model = create_nn_model(input_dim)

    # Le modèle doit avoir un input_dim cohérent
    assert model.input_shape[1] == input_dim
    # Et avoir les méthodes clés
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_train_model():
    model, hist, data = train_model_for_test()
    epochs = data['epochs']

    # Vérifier que l’historique contient bien 'loss' avec le bon nb de valeurs
    assert "loss" in hist.history
    assert len(hist.history["loss"]) == epochs


def test_predict():
    model, hist, data = train_model_for_test()
    X_test = data['X_test']
    y_test = data['y_test']

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    # Vérifier que le MSE est bien un nombre fini
    assert np.isfinite(mse)
