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

        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce").fillna(0)
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce").fillna(0)

        if "family" not in self.df.columns:
            self.df["family"] = "unknown"
        else:
            self.df["family"] = self.df["family"].astype(str).str.lower().str.strip()

  
    def split_multi(self, val):
        if pd.isna(val):
            return set()
        return set(str(val).lower().split(";"))

    def norm(self, val):
        if val is None or val == "":
            return None
        return str(val).strip().lower().replace(" ", "_")

  
    def recommend(self, user_profile=None):

        df = self.df.copy()

        if user_profile is None:
            user_profile = {}

        community = self.norm(user_profile.get("community"))
        category = self.norm(user_profile.get("category"))
        min_price, max_price = user_profile.get("price_range", (None, None))

        df["comm_set"] = df["community"].apply(self.split_multi)
        df["cat_set"] = df["category"].apply(self.split_multi)

        df["explanation"] = ""

        # PRICE MATCH
     
        if min_price is not None and max_price is not None:
            df["price_match"] = df["price"].between(min_price, max_price)
        else:
            df["price_match"] = True

        # COMMUNITY-ONLY CASE
        if community and not category:

            df = df[df["comm_set"].apply(lambda x: community in x)].copy()

            garments = df[df["family"] == "garment"]
            accessories = df[df["family"].isin(["accessory", "jewelry"])]

            result = []
            seen_ids = set()

            def add_items(data, limit):
                count = 0
                for _, r in data.sort_values(by="rating", ascending=False).iterrows():
                    if r["product_id"] not in seen_ids:
                        result.append(r)
                        seen_ids.add(r["product_id"])
                        count += 1
                    if count == limit:
                        break

            add_items(garments, 3)
            add_items(accessories, 2)

            if len(result) < 5:
                fallback = df.sort_values(by="rating", ascending=False)
                for _, r in fallback.iterrows():
                    if r["product_id"] not in seen_ids:
                        result.append(r)
                        seen_ids.add(r["product_id"])
                    if len(result) == 5:
                        break

            final = pd.DataFrame(result)

            final["explanation"] = "community-only diversified rule applied"
            final["score"] = 0   # ✅ FIX ADDED


        # GENERAL CASE
        else:

            def score_row(r):
                score = 0
                reasons = []

                comm = community and (community in r["comm_set"])
                cat = category and (category in r["cat_set"])
                price = r["price_match"]

                if community and category:

                    if comm and cat and price:
                        score += 100
                        reasons.append("community + category + price match")

                    elif comm and cat:
                        score += 85
                        reasons.append("community + category match")

                    elif comm:
                        score += 70
                        reasons.append("community match")

                    elif cat:
                        score += 60
                        reasons.append("category match")

                    elif price:
                        score += 40
                        reasons.append("price match")

                elif community:

                    if comm:
                        score += 100
                        reasons.append("community match")

                elif category:

                    if cat:
                        score += 100
                        reasons.append("category match")

                else:
                    score += 10
                    reasons.append("fallback general")

                return pd.Series([score, " | ".join(reasons)])

            df[["score", "explanation"]] = df.apply(score_row, axis=1)

            df = df[df["score"] > 0].copy()

            df = df.sort_values(by=["score", "rating"], ascending=[False, False])

            if not df.empty:
                top_family = df.iloc[0]["family"]

                df["family_boost"] = df["family"].apply(
                    lambda x: 1 if x == top_family else 0
                )

                df = df.sort_values(
                    by=["score", "family_boost", "rating"],
                    ascending=[False, False, False]
                )

            final = df.head(5)

        if "score" not in final.columns:
            final["score"] = 0

        if "explanation" not in final.columns:
            final["explanation"] = ""


        # OUTPUT 

        cols = [
            "product_id",
            "name",
            "category",
            "community",
            "family",
            "price",
            "rating",
            "image_path",
            "score",
            "explanation"
        ]

        return final[cols]
