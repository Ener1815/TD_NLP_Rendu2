from pydoc import text

import click
import numpy as np
from sklearn.model_selection import cross_val_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model
import pickle
import string
@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task, train_mode=True, vectorizer_path="vectorizer/vectorizer.pkl")
    
    model = make_model()
    model.fit(X, y)
    
    with open(model_dump_filename, 'wb') as f:
        pickle.dump(model, f)



@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(task, model_dump_filename, input_filename, output_filename):
    df = make_dataset(input_filename)
    X = make_features(df, task, train_mode=False, vectorizer_path="vectorizer/vectorizer.pkl")[0]  # Only extract X
    
    with open(model_dump_filename, 'rb') as f:
        model = pickle.load(f)
    
    predictions = model.predict(X)
    
    df['predictions'] = predictions
    df.to_csv(output_filename, index=False)
    

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    if task == "is_comic_video" :

        # Read CSV
        df = make_dataset(input_filename)

        # Make features (tokenization, lowercase, stopwords, stemming...)
        X, y = make_features(df, task, train_mode=True, vectorizer_path="vectorizer/vectorizer.pkl")


        model = make_model()


        # Run k-fold cross validation. Print results
        return evaluate_model(model, X, y)
    else :

        df = make_dataset(input_filename)
        features_list, labels_list, tokens_list = make_features(df, task)

        # Transformez features_list en une matrice 2D
        features_matrix = np.array([list(d.values()) for d in features_list])

        model = make_model()

        # Entraînez le modèle avec les caractéristiques générées
        model.fit(features_matrix, labels_list)

        # Exécutez la validation croisée et imprimez les résultats
        return evaluate_model(model, features_matrix, labels_list)





def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)




if __name__ == "__main__":
    cli()
 