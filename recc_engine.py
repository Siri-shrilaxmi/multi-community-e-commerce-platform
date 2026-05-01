import pandas as pd

class RecommendationEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    # -----------------------------
    # SPLIT HELPERS
    # -----------------------------
    def split_vals(self, val):
        if pd.isna(val):
            return []
        return str(val).replace(";", " ").split()

    # -----------------------------
    # SIMILARITY FUNCTIONS
    # -----------------------------
    def similarity(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        if len(set1) == 0:
            return 0
        return len(set1 & set2) / len(set1)

    # -----------------------------
    # MAIN FUNCTION
    # -----------------------------
    def recommend(self, user_profile, top_n=10):

        community = user_profile["community"]
        preferred_categories = user_profile["preferred_categories"]
        price_min, price_max = user_profile["price_range"]
        past_purchases = user_profile["past_purchases"]
        preferred_occasions = user_profile.get("occasions", [])
        preferred_colors = user_profile.get("colors", [])

        # -----------------------------
        # 1. COMMUNITY FILTER
        # -----------------------------
        df = self.df[self.df["community"] == community].copy()

        # -----------------------------
        # 2. PRICE FILTER
        # -----------------------------
        df = df[(df["price"] >= price_min) & (df["price"] <= price_max)]

        if df.empty:
            return []

        # -----------------------------
        # 3. USER PROFILE FROM HISTORY
        # -----------------------------
        past_df = self.df[self.df["product_id"].isin(past_purchases)]

        user_tags = []
        user_occasions = []

        for _, row in past_df.iterrows():
            user_tags += self.split_vals(row["tags"])
            user_occasions += self.split_vals(row["occasion"])

        # -----------------------------
        # 4. SCORING
        # -----------------------------
        scores = []
        explanations = []

        for _, row in df.iterrows():

            item_tags = self.split_vals(row["tags"])
            item_occasions = self.split_vals(row["occasion"])

            # --- similarities ---
            tag_score = self.similarity(user_tags, item_tags)
            past_score = tag_score

            occasion_score = self.similarity(user_occasions, item_occasions)

            category_score = 1 if row["category"] in preferred_categories else 0

            color_score = 1 if row["color"] in preferred_colors else 0

            rating_score = row["rating"] / 5

            # --- final score ---
            final_score = (
                0.30 * tag_score +
                0.20 * occasion_score +
                0.15 * category_score +
                0.15 * past_score +
                0.10 * rating_score +
                0.10 * color_score
            )

            # -----------------------------
            # EXPLANATION ENGINE 🔥
            # -----------------------------
            reason = []

            if tag_score > 0:
                reason.append("similar to your past style")

            if occasion_score > 0:
                reason.append("fits your occasions")

            if category_score:
                reason.append("matches preferred category")

            if color_score:
                reason.append("matches your color preference")

            if rating_score > 0.8:
                reason.append("highly rated")

            explanation = ", ".join(reason) if reason else "general recommendation"

            scores.append(final_score)
            explanations.append(explanation)

        df["score"] = scores
        df["explanation"] = explanations

        # -----------------------------
        # 5. SORT
        # -----------------------------
        df = df.sort_values(by="score", ascending=False)

        return df.head(top_n)[[
            "product_id", "name", "category", "price",
            "rating", "color", "score", "explanation", "image_path"
        ]]
engine = RecommendationEngine("csv1.csv")

user_profile = {
    "community": "hindu",
    "preferred_categories": ["saree"],
    "price_range": (1000, 3000),
    "past_purchases": [1],
    "occasions": ["festival", "cultural"],
    "colors": ["red", "gold"]
}

result = engine.recommend(user_profile)

print(result)
