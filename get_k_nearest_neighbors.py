from sentence_transformers import SentenceTransformer
import scipy
import pickle
from tqdm import tqdm

def get_closest_neighbors(dataset_text):
    sentence_embeddings = model.encode(dataset_text)
    all_similarities = []
    for i in tqdm(range(len(dataset_text))):
        query = dataset_text[i]
        query_embedding = sentence_embeddings[i]
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        results = [item[0] for item in results]
        assert results[0] == i
        results = results[1:]
        all_similarities.append(results)
    return all_similarities

if __name__ == '__main__':
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    val_data = 'data/paper_yahoo_split/yahoo_val_200_per_class.pkl'
    val_data = pickle.load(open(val_data, 'rb'))
    X = val_data['X']
    get_closest_neighbors(X)


# queries = [query]
# query_embeddings = model.encode(queries)

# # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
# number_top_matches = 3 #@param {type: "number"}

# print("Semantic Search Results")

# for query, query_embedding in zip(queries, query_embeddings):
#     distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

#     results = zip(range(len(distances)), distances)
#     results = sorted(results, key=lambda x: x[1])

#     print("\n\n======================\n\n")
#     print("Query:", query)
#     print("\nTop 5 most similar sentences in corpus:")

#     for idx, distance in results[0:number_top_matches]:
#         print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))