# Online Retail Big Data Project

### 1.1 Business Objective
Identify high-value customers for targeted retention marketing to maximize revenue from the company's most profitable customer segments, addressing the challenge of limited marketing budget in a resource-constrained e-commerce environment.

### 1.2 Analytical Question
"Which customers exhibit high-value purchasing behavior (defined by RFM metrics: Recency, Frequency, Monetary value) and can be grouped into distinct segments for prioritized marketing campaigns, using only transactional data from Invoice, Customer ID, InvoiceDate, Quantity, and Price columns?"

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


### 2.1 Architecture Diagram
![Architecture Diagram](diagram/diagram.png)

