import pandas as pd

class RecommendationEngine:

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # CLEAN DATA
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                )

    # -----------------------------
    # MAIN FUNCTION (SMART RANKING)
    # -----------------------------
    def recommend(self, user_profile=None):

        df = self.df.copy()

        # DEFAULT
        if user_profile is None:
            user_profile = {
                "community": None,
                "category": None,
                "price_range": (None, None)
            }

        def normalize(val):
            if val is None or val == "":
                return None
            return val.strip().lower().replace(" ", "_")

        community = normalize(user_profile.get("community"))
        category = normalize(user_profile.get("category"))
        min_price, max_price = user_profile.get("price_range", (None, None))

        print("\n--- USER INPUT ---")
        print("Community:", community)
        print("Category:", category)
        print("Min:", min_price, "Max:", max_price)

        # -----------------------------
        # SCORE CALCULATION
        # -----------------------------
        df["score"] = 0

        # CATEGORY → highest priority
        # -----------------------------
        # CATEGORY → SMART PRIORITY
        # -----------------------------
        if category:

            # EXACT match → highest
            df.loc[df["category"] == category, "score"] += 5

            # PARTIAL match → lower
            df.loc[
                (df["category"].str.contains(category, na=False)) &
                (df["category"] != category),
                "score"
            ] += 2


        # COMMUNITY → medium priority
        if community:
            df.loc[df["community"] == community, "score"] += 2
        elif not category:
            # only if NOTHING selected → allow "all"
            df.loc[df["community"] == "all", "score"] += 1

        # PRICE → lowest priority
        if min_price is not None and max_price is not None:
            df.loc[
                (df["price"] >= min_price) &
                (df["price"] <= max_price),
                "score"
            ] += 1

        # -----------------------------
        # FILTER OUT ZERO SCORE (OPTIONAL)
        # -----------------------------
        filtered = df[df["score"] > 0]

        # -----------------------------
        # FALLBACK LOGIC
        # -----------------------------
        if filtered.empty:
            print("⚠️ No strong match → fallback to ALL")

            filtered = self.df[self.df["community"] == "all"].copy()

            # if still empty (rare)
            if filtered.empty:
                print("⚠️ No ALL → random items")
                filtered = self.df.sample(min(10, len(self.df)))

        # -----------------------------
        # SORT BY SCORE + RATING
        # -----------------------------
        filtered = filtered.sort_values(
            by=["score", "rating"],
            ascending=[False, False]
        )

        print("✅ Results:", len(filtered))

        return filtered.head(5)


# -----------------------------
# TERMINAL TEST
# -----------------------------
if __name__ == "__main__":

    engine = RecommendationEngine("csv1.csv")

    test_profile = {
        "community": "hindu",
        "category": "salwar suit",
        "price_range": (2000, 5000)
    }

    results = engine.recommend(test_profile)

    print("\n=== FINAL RESULTS ===")
    print(results[[
        "product_id",
        "name",
        "category",
        "community",
        "price",
        "rating",
        "score"
    ]])
