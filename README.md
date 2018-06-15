# Kaggle\_HomeCreditDefaultRisk
Repo for Home Credit Default Risk competition on Kaggle

## Data Description 

- <b>application\_{train|test}.csv:</b> This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).  Static data
  for all applications. One row represents one loan in our data sample.

- <b>bureau.csv:</b> All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have
a loan ur sample).  For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application.

- <b>bureau\_balance.csv:</b> Monthly balances of previous credits in Credit Bureau.  This table has one row for each month of history of every
  previous it reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some
history rvable for the previous credits) rows.

- <b>POS\_CASH\_balance.csv:</b> Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
  This e has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample
– the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) .

- <b>credit\_card\_balance.csv:</b> Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.  This table has one
  row for month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table
has ans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

- <b>previous\_application.csv:</b> All previous applications for Home Credit loans of clients who have loans in our sample.  There is one row for
  each ious application related to loans in our data sample.

- <b>installments\_payments.csv:</b> Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.  There
  is a) row for every payment that was made plus b) one row each for missed payment.  One row is equivalent to one payment of one installment OR one
allment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

- <b>HomeCredit\_columns\_description.csv:</b> This file contains descriptions for the columns in the various data files.

