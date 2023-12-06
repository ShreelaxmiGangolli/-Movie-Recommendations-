import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')


movie_ratings = pd.merge(ratings_data, movies_data, on='movieId')


print("Merged Dataset:")
print(movie_ratings.head())


user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')


user_movie_matrix = user_movie_matrix.fillna(0)


user_similarity = cosine_similarity(user_movie_matrix)


user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)


def get_movie_recommendations(user_id, top_n=5):
    user_row = user_similarity_df[user_id]
    similar_users = user_row.sort_values(ascending=False).index[1:]
    
    
    unrated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] == 0].index
    
    
    predicted_ratings = user_movie_matrix.loc[similar_users, unrated_movies].mean(axis=0)
    
    
    recommended_movies = predicted_ratings.sort_values(ascending=False).head(top_n)
    
    return recommended_movies


user_id = 1
recommendations = get_movie_recommendations(user_id, top_n=5)

print(f"\nTop 5 Movie Recommendations for User {user_id}:")
print(recommendations)
 