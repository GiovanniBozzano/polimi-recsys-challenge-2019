import copy
import os
import pandas as pd
import scipy.sparse as sparse
from datetime import datetime
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import mean_average_precision_at_k, train_test_split


def create_csv(results, results_dir='./'):
    csv_filename = 'results_'
    csv_filename += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'
    with open(os.path.join(results_dir, csv_filename), 'w') as file:
        file.write('user_id,item_list\n')
        for key, value in results.items():
            file.write(str(key) + ',')
            first = True
            for prediction in value:
                if not first:
                    file.write(' ')
                first = False
                file.write(str(prediction))
            file.write('\n')


data = pd.read_table('dataset/data_train.csv', delimiter=',')
data.columns = ['user', 'item', 'bought']

# The implicit library expects data as a item-user matrix so we
# create two matrices, one for fitting the model (item-user)
# and one for recommendations (user-item)
sparse_item_user = sparse.csr_matrix((data['bought'].astype(float), (data['item'], data['user'])))
sparse_user_item = sparse.csr_matrix((data['bought'].astype(float), (data['user'], data['item'])))

train, test = train_test_split(sparse_item_user, train_percentage=0.75)

# factors (int): How many latent features we want to compute.
# regularization (float): Regularization value
# iterations (int): How many times we alternate between fixing and updating our user and item vectors

# Initialize the als model and fit it using the sparse item-user matrix
model = AlternatingLeastSquares(factors=256, regularization=1.0, iterations=5, calculate_training_loss=True)

# alpha_val (int): The rate in which we'll increase our confidence in a preference with more interactions.
# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 25
data_confidence = (train * alpha_val).astype('double')

epochs = 50
# Fit the model
model = copy.deepcopy(model)
for epoch in range(int(epochs / 5)):
    model.fit(data_confidence)
    p = mean_average_precision_at_k(model, train.T.tocsr(), test.T.tocsr(), K=10, num_threads=8)
    print(p)

to_predict = pd.read_table('dataset/data_target_users_test.csv', delimiter=',')
to_predict.columns = ['user']

results = {}

for user in to_predict['user']:

    # Use the implicit recommender
    recommended = model.recommend(user, sparse_user_item)

    predicted_items = []
    predicted_scores = []

    # Get predicted items and scores
    for item in recommended:
        idx, score = item
        predicted_items.append(idx)
        predicted_scores.append(score)

    # Create a dataframe of items and scores
    recommendations = pd.DataFrame({'item': predicted_items, 'score': predicted_scores})
    results[user] = predicted_items

create_csv(results)
