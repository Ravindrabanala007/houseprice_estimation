
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("data/house_data.csv")

plt.figure(figsize=(8,5))

plt.scatter(
    data["living area"],
    data["Price"]
)

plt.xlabel("Living Area")
plt.ylabel("Price")
plt.title("Living Area vs Price")

# plt.savefig(
#     "screenshots/living_area_vs_price.png"
# )


plt.show()


plt.figure(figsize=(12,8))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

# plt.savefig(
#     "screenshots/correlation_heatmap.png"
# )



plt.show()


plt.figure(figsize=(8,5))

sns.histplot(
    data["Price"],
    bins=50,
    kde=True
)

plt.title("Price Distribution")
plt.savefig(
    "screenshots/price_distribution.png"
    
)


plt.show()




plt.figure(figsize=(8,5))

sns.boxplot(
    x=data["No of bedrooms"],
    y=data["Price"]
)

plt.title("Bedrooms vs Price")

# plt.savefig(
#     "screenshots/bedrooms_vs_price_boxplot.png"
# )

plt.show()