import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Set a more attractive and modern style for the plots
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Load the dataset
df = pd.read_csv('flipkart_sales_analysis/flipkart_com-ecommerce_sample.csv')

# Initial data exploration
print('Shape of the data:', df.shape)
print('\nFirst 5 rows of the data:')
print(df.head())
print('\nColumns in the data:')
print(df.columns)

# Data cleaning
# Drop unnecessary columns
df.drop(['uniq_id', 'crawl_timestamp', 'product_url', 'pid', 'image', 'is_FK_Advantage_product', 'overall_rating', 'product_specifications'], axis=1, inplace=True)

# Handle missing values and data types
df.dropna(subset=['brand'], inplace=True) # Drop rows with no brand
df['description'].fillna('', inplace=True) # Fill missing descriptions

# Clean and convert price columns
df['retail_price'] = pd.to_numeric(df['retail_price'], errors='coerce')
df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
df.dropna(subset=['retail_price', 'discounted_price'], inplace=True)
df = df[df['retail_price'] > 0] # Avoid division by zero for discount calculation

# Clean and convert rating column
df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
df.dropna(subset=['product_rating'], inplace=True)

# Category analysis
df['product_category_tree'] = df['product_category_tree'].apply(lambda x: x.split('>>')[0][2:])
category_counts = df['product_category_tree'].value_counts().head(10)

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Top 10 Most Populated Product Categories', fontsize=16)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Number of Products', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=10, padding=3)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/top_10_categories.png', dpi=150)


# Brand analysis
brand_counts = df['brand'].value_counts().head(10)

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=brand_counts.index, y=brand_counts.values, palette='plasma')
plt.title('Top 10 Brands by Product Count', fontsize=16)
plt.xlabel('Brand', fontsize=12)
plt.ylabel('Number of Products', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=10, padding=3)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/top_10_brands.png', dpi=150)


# Price analysis
plt.figure(figsize=(12, 7))
sns.histplot(df['discounted_price'], bins=50, kde=True, color='#2c7fb8')
plt.title('Distribution of Discounted Prices', fontsize=16)
plt.xlabel('Discounted Price (INR)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/price_distribution.png', dpi=150)


# Customer behavior analysis
plt.figure(figsize=(12, 7))
sns.scatterplot(x='discounted_price', y='product_rating', data=df, alpha=0.6, edgecolor=None, color='#d95f02')
plt.title('Price vs. Product Rating', fontsize=16)
plt.xlabel('Discounted Price (INR)', fontsize=12)
plt.ylabel('Product Rating (out of 5)', fontsize=12)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/price_vs_rating.png', dpi=150)


# Unique insights
df['description_length'] = df['description'].str.len()

plt.figure(figsize=(12, 7))
sns.scatterplot(x='description_length', y='product_rating', data=df, alpha=0.6, edgecolor=None, color='#7570b3')
plt.title('Description Length vs. Product Rating', fontsize=16)
plt.xlabel('Length of Product Description (characters)', fontsize=12)
plt.ylabel('Product Rating (out of 5)', fontsize=12)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/description_length_vs_rating.png', dpi=150)

# Another unique insight: What is the average discount percentage for each category?
df['discount_percentage'] = ((df['retail_price'] - df['discounted_price']) / df['retail_price']) * 100
category_discount = df.groupby('product_category_tree')['discount_percentage'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=category_discount.index, y=category_discount.values, palette='magma')
plt.title('Top 10 Categories by Average Discount', fontsize=16)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Average Discount (%)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f%%', label_type='edge', fontsize=10, padding=3)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/category_discounts.png', dpi=150)


# Sentiment Analysis of Product Descriptions
df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

plt.figure(figsize=(12, 7))
sns.scatterplot(x='sentiment', y='product_rating', data=df, alpha=0.6, edgecolor=None, color='#1b9e77')
plt.title('Description Sentiment vs. Product Rating', fontsize=16)
plt.xlabel('Sentiment Polarity of Description', fontsize=12)
plt.ylabel('Product Rating (out of 5)', fontsize=12)
plt.tight_layout()
plt.savefig('flipkart_sales_analysis/plots/sentiment_vs_rating.png', dpi=150)


# Save the cleaned data
df.to_csv('flipkart_sales_analysis/cleaned_data.csv', index=False)

print('\nCleaned data saved to cleaned_data.csv')
