from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, corr
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Mental Health Data Analysis") \
    .getOrCreate()

# Load the CSV file
df = spark.read.csv("/opt/spark/mental_health_dataset.csv", header=True, inferSchema=True)

# Show the initial dataframe structure
df.show(5)
df.printSchema()

# Preprocessing
# Handle missing values
df = df.na.fill({"self_employed": "Unknown"})
df = df.na.drop()

# Standardize and encode categorical variables
categorical_cols = ["Gender", "self_employed", "family_history", "treatment", "work_interfere"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_vec") for col in categorical_cols]

# Feature selection
selected_cols = ["Gender_vec", "self_employed_vec", "family_history_vec", "treatment_vec", "work_interfere_vec", "Age"]

# Assemble features
assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")

# Create and fit the pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler])
model = pipeline.fit(df)
df_encoded = model.transform(df)

# Convert to Pandas for easier analysis
pdf = df_encoded.toPandas()

# Descriptive Statistics
print("Descriptive Statistics:")
print(pdf.describe())

# Correlation Analysis
correlation_matrix = df_encoded.select([col(c).cast("double") for c in selected_cols]).toPandas().corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix of Mental Health Variables")
plt.tight_layout()
plt.savefig("/opt/spark/correlation_matrix.png")

# Chi-Square Test
chi_square = ChiSquareTest.test(df_encoded, "features", "treatment_index")
print("\nChi-Square Test Results:")
print(chi_square.select("pValues").show())

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="treatment_index")
lr_model = lr.fit(df_encoded)
print("\nLogistic Regression Coefficients:")
print(lr_model.coefficients)

# Save preprocessed data
df_encoded.write.parquet("/opt/spark/preprocessed_mental_health_data.parquet")

# Stop the Spark session
spark.stop()
