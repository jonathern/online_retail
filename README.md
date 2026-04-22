# Online Retail Big Data Project

### 1.1 Business Objective
Identify high-value customers for targeted retention marketing to maximize revenue from the company's most profitable customer segments, addressing the challenge of limited marketing budget in a resource-constrained e-commerce environment.

### 1.2 Analytical Question
"Which customers exhibit high-value purchasing behavior (defined by Recency, Frequency, Monetary value) and can be grouped into distinct segments for prioritized marketing campaigns, using only transactional data from Invoice, Customer ID, InvoiceDate, Quantity, and Price columns?"

### 1.3 Expected Outcomes
#### Deliverables
- Customer segmentation report with 3-5 RFM-based clusters (High-Value, Loyal, At-Risk, Lost, Price-Sensitive)
- Top 20% high-value customer list with actionable contact data (CustomerID, Country)
- Segment performance metrics (average revenue per segment, retention rate, purchase frequency)

#### Success Metrics
- Silhouette Score > 0.5 for cluster quality
- 80/20 Pareto validation: Top 20% customers generate ≥80% revenue
- Business ROI proxy: High-value segment shows 3x higher lifetime value vs average

#### Stakeholder
Marketing Manager who receives a prioritized customer list for email/SMS retention campaigns, enabling 20% budget allocation to customers generating 80% revenue.

#### Practical Use
Marketing team imports the "High-Value Customer List" CSV into their CRM/email platform to execute personalized discount campaigns, expecting 15-25% uplift in repeat purchase rate within 90 days.

## TASK 2: BIG DATA ARCHITECTURE DESIGN 

### 2.1 Architecture Diagram
![Architecture Diagram](diagram/diagram.png)

### 2.2 Layer Architecture Description and Justification
The complete architecture is batch, on-premise, and depends on open-source tooling. All processing runs on a Macbook with no internet dependency.
#### Raw Data Layer
Tools: Excel (.xlsx), CSV files on local disk
Processing Mode : Batch 
Role: Acts as the primary cource of transactional records, storing transactional data (Invoice, Customer ID, Quantity, Price, InvoiceDate, Country)
Justification:The source data was provided in Excel format, no cloud storage is needed. No transformation of the source format is needed at this layer. Keeping raw files untouched preserves an audit trail and allows full re-runs from the original data at any point.

### Ingestion Layer
Tools: pandas.read_excel() + openpyxl, spark.createDataFrame()
Role: Reads raw Excel and CSV files from disk and converts them into a PySpark DataFrame. Basic schema inference runs at this stage, and rows with malformed date fields or missing critical columns are logged and dropped before passing downstream.
Justification: Pandas handles Excel parsing reliably since PySpark has no native Excel reader without additional plugins. The DataFrame is handed to Spark immediately after loading to avoid keeping large data in Pandas memory longer than necessary. For this dataset size, the Pandas-to-Spark handoff stays within single-node memory limits. If the dataset grows beyond available RAM, this layer would need replacing with a direct Spark CSV reader or a dedicated Excel-to-Parquet conversion step first.

### Pre-Processing Layer
Tools: Python (language),PySpark(processing framework), Jupyter Notebook(execution environment), 
Processing mode: Batch
Role: This layer handles four sequential operations. First, data cleaning removes duplicate rows and filters out Nulls from Customer ID, Quantity, Price, and drops rows where Quantity or Price are zero or negative. 
Second, RFM feature engineering computes Recency (days since last purchase relative to the most recent date in the dataset), Frequency (count of distinct invoices per customer), and Monetary (sum of Quantity multiplied by Price per customer). Customers with no purchase activity in the last 365 days are excluded at this stage. 
Third, outlier detection converts the RFM table to Pandas, applies IQR bounds independently across Recency, Frequency, and Monetary, and removes customers falling outside those bounds before converting back to a Spark DataFrame. 
Fourth, feature assembly and scaling use VectorAssembler to combine the three RFM columns into a single feature vector, then StandardScaler normalises the vector to prevent high Monetary values from distorting KMeans distance calculations.
Justification: Separating cleaning, feature engineering, outlier removal, and scaling as distinct sequential steps makes each independently testable and reproducible. The IQR method was chosen for outlier removal because it is non-parametric and does not assume a normal distribution, which RFM data rarely follows. Converting to Pandas for IQR is acceptable at this dataset size since the RFM table holds one row per customer, not one row per transaction.

### Storage Layer
Tools: Parquet files, SQLite
Role: Persists raw data, RFM features, and final customer segments for later audit/re-run, or analysis further down the line.
Justification: Parquet files feature better compression compared to CSV, while SQLite offers lightweight relational store for segment lookup.

### Analytics Layer
Tools: PySpark MLlib (StandardScaler, KMeans), groupBy().count()
Role: K-means clustering on RFM features, silhouette score validation, segment profiling
Justification: MLlib included in PySpark—no extra installs. Scales to full dataset while running locally.

### Serving Layer
Tools: Excel files written via PySpark
Processing mode: Batch
Role: Delivers segments of interest to the Marketing Manager including the top segment “Champions” for targeted marketing campaigns. The file contains one row per customer with their RFM scores and segment label. 
Justification: Excel is used universally and is compatible with CRM tools, with no cloud dependency.

### Conclusion
The pipeline is achieved 100% locally on-device, with all components running on a single MacBook Pro. No intenet connection, cloud resources, and no local servers are required. All of the tooling used is open-source and requires zero extra funds or budget. One Jupyter notebook orchestrates everything.

## Task 3: Data Processing Strategy
### 3.1 Batch Processing Strategy
#### i. Historical data used
All transactions in the file online_retail_II.xlsx is used, spanning the full time range of InvoiceDate (historical snapshot of customer behavior).

#### ii. Computation performed

```
# Complete RFM Batch Job
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans

# RFM Calculation (Historical Analysis)
rfm = (df
    .filter(col("CustomerID").isNotNull())
    .groupBy("CustomerID", "Country")
    .agg(
        datediff(max("InvoiceDate"), current_date()).alias("Recency"),  # Days since last purchase
        countDistinct("InvoiceNo").alias("Frequency"),                  # Unique invoices
        round(sum(col("Quantity") * col("Price")), 2).alias("Monetary") # Total spend
    )
)

# K-Means Clustering on RFM features
assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
kmeans = KMeans(k=5, seed=42)

# Pipeline execution
scaled_rfm = scaler.fit(assembler.transform(rfm)).transform(assembler.transform(rfm))
clusters = kmeans.fit(scaled_rfm)
rfm_segments = clusters.transform(scaled_rfm)

# Export High-Value Segment
high_value = rfm_segments.filter(col("prediction") == 0)  # High-value cluster
high_value.select("CustomerID", "Country", "Recency", "Frequency", "Monetary").coalesce(1).write.csv("high_value_customers.csv")
```

#### iii. Why batch processing is appropriate or inappropriate
Batch processing is more appropriate for this scenario because:
- Historical RFM analysis requires complete dataset, without partial updates.
- Customer segmentation is computed weekly/monthly, and not in real-time.
- Considering the single laptop constraint, batch processing maximizes the resource utilization.
- Batch processing is offline-first, requiring no streaming infrastructure, thus saving costs.
- It is okay in the context of a marketing campaign to use static lists (CSV export), and not live updates.

#### iv. Streaming processing
Streaming processing is not implemented, but rather simulated in this section. For this approach, the input stream are new transactions, recorded to a directory we shall call `data/streaming/`. The output is a live updating customer segment dashboard, as well as any supporting details that the Marketing Manager would require, for example, a CSV of the top segment to target for marketing activity.

##### Simulated streaming code
```
# Simulated Structured Streaming (new transactions folder)
streaming_df = (spark
    .readStream
    .format("csv")
    .option("header", "true")
    .schema(existing_schema)  # From batch job
    .load("data/streaming/")
)

# Live RFM updates (windowed aggregation)
rfm_stream = (streaming_df
    .filter(col("CustomerID").isNotNull())
    .groupBy("CustomerID", window("InvoiceDate", "1 day"))
    .agg(sum(col("Quantity") * col("Price")).alias("Monetary"))
)

query = (rfm_stream
    .writeStream
    .outputMode("complete")
    .format("console")
    .start())
```

In the end, streaming processing is not suited to this scenario for the following reasons:
- There is no requirement for real time data, as the marketing team only requires weekly updates and sufficient time between updates to target customers and implement a marketing strategy.
- Considering that there is an infrastructure limitation (only a single laptop is available), we cannot use modern streaming tools such as Kafka or Flume.
- There is a further constraint, in the form of the requirement to perform the analysis and segmentation in an offline environment on the laptop. In this scenario, streaming processing is not a realistic choice.

### 3.2 Trade-off Analysis

| Dimension | Batch (Chosen) | Streaming (Not Chosen) |
|----------|----------|----------|
| Latency   | Weekly updates are acceptable in this context | Milliseconds, which is overkill for this context  |
| Cost   | $0  | $0, but unlikely to be eliable at this price point |
| Complexity | Simple, requiring a single python notebook | Complex, potentially requiring multiple cloud services |
| Accuracy |  |  |

For our chosen scenario, RFM segmentation requires a complete historical context, and our Marketing Manager needs customer lists weekly, rather than live updates. Batch processing delivers identical business value with far less complexity on our single MacBook Pro setup. Therefore, batch processing is the better approach, than streaming.

## Task 4: 