import pandas as pd
import sys
import scipy.stats as stats
import numpy as np
from NISQA_lib import calc_eval_metrics,fit_first_order,calc_mapped
import matplotlib.pyplot as plt

# Read in the data
def calc_rmse(y_true, y_pred, d=0):
    if d==0:
        rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    else:
        N = y_true.shape[0]
        if (N-d)<1:
            rmse = np.nan
        else:
            rmse = np.sqrt( 1/(N-d) * np.sum( np.square(y_true-y_pred) ) )  # Eq (7-29) P.1401
    return rmse





in_data = pd.read_csv(sys.argv[1])
y = in_data["mos"]
y_hat= in_data["predicted_mos"]

b = fit_first_order(y,y_hat)
d = 1
y_hat_mapped = calc_mapped(y_hat, b)

r = calc_eval_metrics(y,y_hat,y_hat_map=y_hat_mapped,d=d)

#print(r)
name = sys.argv[1].split("/")[-1].strip(".csv")
r_p = r["r_p"]
r_s = r["r_s"]
rmse = r["rmse"]
rmse_map = r["rmse_map"]
#print("%s,%f%,%f,%f"%(name,r_p,rmse,rmse_map))
print("%s,%f,%f,%f,%f"%(name,r_p,r_s,rmse,rmse_map))
print("--------")
plt.suptitle(sys.argv[1].split("/")[-1].strip(".csv")) 
plt.subplot(1,2,1)
plt.scatter(y,y_hat)
plt.xlabel("mos")
plt.ylabel("predicted_mos")
plt.xlim([1,5])
plt.ylim([1,5])
plt.subplot(1,2,2)
plt.scatter(y,y_hat_mapped)
plt.xlabel("mos")
plt.xlim([1,5])
plt.ylim([1,5])
plt.ylabel("predicted_mos_mapped")
plt.tight_layout()
plt.savefig(sys.argv[1].strip(".csv")+".png")