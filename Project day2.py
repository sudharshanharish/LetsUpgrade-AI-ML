{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from datetime import datetime as dt\n",
    "from statsmodels.tsa.stattools import adfuller, acf, pacf\n",
    "from statsmodels.tsa.arima_model import ARIMA\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import pandas.tseries\n",
    "from matplotlib.pylab import rcParams\n",
    "rcParams['figure.figsize'] = 15, 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset=pd.read_csv(r'C:\\Users\\User\\Downloads\\Airpassengers.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Month</th>\n",
       "      <th>#Passengers</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1949-01</td>\n",
       "      <td>112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1949-02</td>\n",
       "      <td>118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1949-03</td>\n",
       "      <td>132</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1949-04</td>\n",
       "      <td>129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1949-05</td>\n",
       "      <td>121</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Month  #Passengers\n",
       "0  1949-01          112\n",
       "1  1949-02          118\n",
       "2  1949-03          132\n",
       "3  1949-04          129\n",
       "4  1949-05          121"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>#Passengers</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Month</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1949-01-15</th>\n",
       "      <td>112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1949-02-15</th>\n",
       "      <td>118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1949-03-15</th>\n",
       "      <td>132</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1949-04-15</th>\n",
       "      <td>129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1949-05-15</th>\n",
       "      <td>121</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            #Passengers\n",
       "Month                  \n",
       "1949-01-15          112\n",
       "1949-02-15          118\n",
       "1949-03-15          132\n",
       "1949-04-15          129\n",
       "1949-05-15          121"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset['Month'] = dataset['Month'].apply(lambda x: dt(int(x[:4]), int(x[5:]), 15))\n",
    "dataset = dataset.set_index('Month')\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "ts = dataset['#Passengers']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x2fb9c1c0c8>]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA20AAAFlCAYAAAB4PgCOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdeXicZ30v/O89u6RZtM1olyVZ8iYvseM4e4CENqwlQEmhhyaFFGhf3sJbaAuHlnPa0tIDLYVC6UIhZSckFJo0bwiEJEA2HK/yJluStY32bUaj2bf7/DHzyHK0jaSZeeYZfz/XlcvSM49Gt2Q51/z0u+/vT0gpQURERERERIVJp/YCiIiIiIiIaHUs2oiIiIiIiAoYizYiIiIiIqICxqKNiIiIiIiogLFoIyIiIiIiKmAs2oiIiIiIiAqYQe0FAEB1dbVsaWlRexlERERERESqOHHixIyU0rnSYwVRtLW0tOD48eNqL4OIiIiIiEgVQoih1R7j9kgiIiIiIqICxqKNiIiIiIiogLFoIyIiIiIiKmAs2oiIiIiIiAoYizYiIiIiIqICxqKNiIiIiIiogLFoIyIiIiIiKmAs2oiIiIiIiAoYizYiIiIiIqICxqKNiIiIiIiogLFoIyIiIiIiKmAs2oiIiIiISDXhWAL90361l1HQWLQREREREZFqvvXSEF77D7/A+bF5tZdSsFi0ERERERGRagZnA0hK4C8eOw8ppdrLKUgs2oiIiIiISDWTvjB0Ajg26MGjp8fUXk5BYtFGRERERESqmfCFcWt7NQ40OvDpJ7rhj8TVXlLBYdFGRERERESqmZiPoN5Rgr/4jU5MLUTwpad71V5SwWHRRkREREREqoglkpgNRFDjsOBgcwXuPdyIB18YQN8U0ySXYtFGRERERESqmFqIQEqg1m4BAPzp63bBYtTjL/+boSRLsWgjIiIiIiJVTMyHAQC1DjMAoNpqxh+9dgee653BTy9Mqrm0gsKijYiIiIiIVDHpSxVtNelOGwDcd/M27Kyx4VOPX0A4llBraQWFRRsREREREalisdO2pGgz6HX437+xByOeEH54clStpRWUjIo2IUS5EOIHQoiLQohuIcTNQohKIcRTQoje9J8V6XuFEOKLQog+IcQZIcSh3H4JRERERESkRZO+MEx6HSrLTFddv7mtCiaDDkNzAZVWVlgy7bT9I4AnpZS7ABwA0A3g4wCellJ2AHg6/T4AvB5AR/q/9wP4l6yumIiIiIiIisKELwyX3QwhxFXXhRBwWs2Y9kVUWllhWbdoE0LYAdwB4GsAIKWMSim9AN4C4Bvp274B4J70228B8E2Z8isA5UKIuqyvnIiIiIiING1iPnzV1silnDYzphZYtAGZddraAEwD+A8hxCkhxFeFEGUAaqSU4wCQ/tOVvr8BgHvJx4+kr11FCPF+IcRxIcTx6enpLX0RRERERESkPVMLqRltK3HZzJhaCOd5RYUpk6LNAOAQgH+RUh4EEMCVrZArEStcWzZkQUr5FSnlYSnlYafTmdFiiYiIiIioOEgp1+y0uexmTLPTBiCzom0EwIiU8mj6/R8gVcRNKtse039OLbm/acnHNwIYy85yiYiIiIioGPjCcYRiidWLNpsFnmAM0XgyzysrPOsWbVLKCQBuIcTO9KW7AFwA8BiA+9PX7gfwaPrtxwDcl06RvAnAvLKNkoiIiIiICFgyo22V7ZFOW2rg9rSf3TZDhvf9IYDvCCFMAPoBvAepgu9hIcQDAIYBvCN97xMA3gCgD0AwfS8REREREdGilWa0LeVKF21TvjAaykvytq5ClFHRJqU8DeDwCg/dtcK9EsAHt7guIiIiIiIqYhO+9Yq21HWea8t8ThsREREREVHWTKY7bS67ecXHle2RjP1n0UZERERERCqY8IVRUWqExahf8fFqqwlCsGgDWLQREREREZEKJn1h1KyyNRIADHodqspM3B4JFm1ERERERKSCCV8YtaskRyqcNgumOWCbRRsREREREeXfxHwENbb1ijYzt0eCRRsREREREeVZLJHEbCCy6ow2hctmxpSPRRuLNiIiIiIiyquphQikXD3uX+GymTHjjyCZlHlaWWFi0UZERERERHm1OFjbsXLcv8JlMyOelPAEo/lYVsFi0UZERERERHk1mR6svVZ6JJAKIgEY+8+ijYiIiIiI8mqx07be9kg7B2wDLNqIiIiIiCjPJn1hmPQ6VJaZ1rzPZUsVbdf6rDYWbURERERElFcTvjBcdjOEEGve57QpnbZre1YbizYiIiIiIsqrifnwulsjAaDUZIDVbLjmY/9ZtBERERERUV5N+sLrzmhTuGxmbo9UewFERERERHTtkFJiwpdZpw1IbZFk0UZERERERJQnvlAc4VhyQ0Ubz7QREREREVHBGZ8PIZ5Iqr2MrJtMF2CZb4+0MPJf7QUQEREREdHVfn5pCrd95ll87+VhtZeSdZnOaFO47GYEown4I/FcLqugsWgjIiIiIiog3eM+/L/fPYVEUmJwNqj2crJuwrfBoo2z2li0EREREREViklfGO/9+jGUmfWotpox6Su+s1yT6U6by27O6P7FWW1F+L3IFIs2IiIiIqICEIjE8d6vH8N8KIYHf/cGtFWXFeVZrglfGBWlRliM+ozud9lSHbli/F5kikUbEREREZHKEkmJDz90Ct3jPnz5tw+hs94Bl704o+4nfWHUZLg1EriyPZJFGxERERERqeZTj1/Az7qn8Je/0YnX7HIBSKcmFuGWwAlfGLUZJkcCQHmpEUa9KMoCNlMs2oiIiIiIVPTkuXF8/cVBPHBbK37n5pbF6y67GYEiTE2cmI9kHEICAEIIOK3X9qw2Fm1ERERERCo6NuiBxajDJ96w+6rrriIM4IglkpgNRDa0PRIAnHYLO21ERERERKSOMW8IDeUl0OvEVdeVwqaYznJNLUQgJTa0PRJIFbBTvuL5PmwUizYiIiIiIhWNekNoqChddr0YAzg2Olhb4bSZMe0vnu/DRrFoIyIiIiJS0agn1Wl7pcWo+yLaHqnMndvo9kiXzYy5QBTReDIXyyp4LNqIiIiIiFQSiiYwG4iisWJ50WYvMcBk0BVnp23D2yNT989co902Fm1ERERERCoZ9YYAAPXly4sYIQRq7Oai67SZDDpUlBo39HHKVtFrNYyERRsRERERkUqUoq2hfPmZNiA9q62ICpUJXxg1djOEEOvfvISzCM/3bQSLNiIiIiIilYx60kXbCtsjgVSHabKIOm1j3hDqHCt/rWtx2ZWirXi+FxvBoo2IiIiISCVj3hD0OoGadCfplVw2c1F1l/qnA2irLtvwx1VblZl1xfO92AgWbUREREREKhn1hlBrt8CgX/lluctuwUI4jnAskeeVZd98MIbZQBRtzo0XbUa9DpVlpms29p9FGxERERGRSkY9oVW3RgJLZrUVQYepf8YPAGittm7q46/lAdss2oiIiIiIVDLqDaFxhRltCld6ntlkEZzlGpgJAMCmOm1AesB2EXwfNoNFGxERERGRCuKJJCZ8YdSvVbQVU6dtOgC9TqCpYuWkzPUUW5LmRrBoIyIiIiJSwYQvjERSrrk9sibdaSuG1MSBmQCaK0thMmyuBHHazJjxR5BMyiyvrPCxaCMiIiIiUsFi3P8anbaKUiOMelEUHabL0360biI5UuGymRFLSHhDsSyuShtYtBERERERqWBxsPYanTYhBJxW7c9qSyYlBmcDWyvaruFZbSzaiIiIiIhUMOZdv9MGAE67BdMa77SN+8IIx5KbDiEBUmfagOI437dRLNqIiIiIiFQw6g2h2mqCxahf875iiLofmE4lR26l0+ZMh7JovYDdDBZtREREREQqGPGE1u2yAUCN3az5LYHKjLbtzs3NaAOWJGmyaCMiIiIionwY9YbWjPtXuGwWeIIxROKJPKwqN/qnAygz6RcLr80oMxtQZtJrvoDdDBZtRERERER5JqXEmDezTpurCLYF9s8E0OosgxBiS8/jtJnZaSMiIiIiotybDUQRjiXXTI5UXElN1G6xMjDjR2v15rdGKlw27YeybAaLNiIiIiKiPMtkRptC66mJ4VgCI54Q2rYQQqJw2rU//mAzWLQRERERUUEamg1gLhBVexk5MZbBjDaF1ueTDc8FISW2FPev6HBZMTwXRCASz8LKtCOjok0IMSiEOCuEOC2EOJ6+VimEeEoI0Zv+syJ9XQghviiE6BNCnBFCHMrlF0BERERExWfEE8Qbv/g8Pv1Et9pLyQllsHZjeem691aVmaET2u209U+nkiPbsrA9cm+9A1IC3eO+LT+Xlmyk0/YaKeV1UsrD6fc/DuBpKWUHgKfT7wPA6wF0pP97P4B/ydZiiYiIiKj4JZMSH/vPM/BH4ovbCIvNiCcEq9kAe4lh3Xv1OoFqq3Zj//tnUjPaWqrXL1DXs7fBAQA4Nzq/5efSkq1sj3wLgG+k3/4GgHuWXP+mTPkVgHIhRN0WPg8RERERXUO+c3QIL/TNwmYxYNqvze7SelJx/5aM0xRr7BbNBpH0Twfgsplhsxi3/Fw1djOqrSacG2OnbSUSwE+FECeEEO9PX6uRUo4DQPpPV/p6AwD3ko8dSV+7ihDi/UKI40KI49PT05tbPREREREVlaHZAD79xEXcscOJe65rKNqkwNEMB2srXDYzJjW6PXJgJoDWLISQAIAQAp31Dpxn0baiW6WUh5Da+vhBIcQda9y70q8L5LILUn5FSnlYSnnY6XRmuAwiIiIiKlaJpMSfPHIGBr3AZ96+Dy6bGfMhbQ+VXs2oN5RRCInCZTdjWqPbIwdmAmhzbv08m2Jvgx29kwsIx4rv52I1GRVtUsqx9J9TAH4E4AiASWXbY/rPqfTtIwCalnx4I4CxbC2YiIiIiIrTf7wwgJcH5/C/39yJOkcJnOmh0jP+4kqQ9EfimA/F0JBBCInCabNgNhBFPJHM4cqyzxuMYi4QzUrcv2JvvQPxpETP5ELWnrPQrVu0CSHKhBA25W0Avw7gHIDHANyfvu1+AI+m334MwH3pFMmbAMwr2yiJiIiIiFbSN+XHZ39yCa/d7cLbD6VO1ihFW7FtkdxI3L+ixm6GlNorYJUQkmzE/SuuhJFcO1sk14+rAWoA/Ch9SNIA4LtSyieFEMcAPCyEeADAMIB3pO9/AsAbAPQBCAJ4T9ZXTURERERFI55I4qOPdKHUpMen37ZvMZyjWIu2jQzWVigDtid9YdQ6LDlZVy70T6eKtmydaQOAxooS2C0GnBu7dhIk1y3apJT9AA6scH0WwF0rXJcAPpiV1RERERFR0XuubwZdbi/+4d4Di8UJULxF24gyo20jZ9psyoBtbX0vBmb8MOgEmiq3HvevEEJgb4MD56+h2P+tRP4TEREREW3ZpYnU2aTX7qm56npVWXEWbaOeEIx6AafVnPHHuOxK0aatMJL+6QCaK0th1Ge37Oist6N7YgExjZ3x2ywWbURERESkqt5JP2rsZthfMcfLZNChotSIab+2CpX1jHpDqHOUQKfLbEYbAFRbzRACmNJY7H824/6X2tvgQDSeRN+UP+vPXYhYtBERERGRqvqmFtDhsq34mNNmLsJOW3BD59kAwKjXoarMpKlOWzIp03H/2S/aOuuVMJJrY4skizYiIiIiUo2UEr1TfrS7Vp7jVZRF2wZntCmcNoumOm1j8yFE4km0VmdvRpuitboMpSb9NTNkm0UbEREREalmbD6MYDSBjppVijarGdN+7RQq64nGk5haiGy40wakwki0FESiJEfmotOm1wnsqbPj/DWSIMmijYiIiIhU05sekLze9shUQLn2TcyHIeXGZrQpauxmTW2PHFBmtOXgTBuQOtd2fsyHZLI4fjbWwqKNiIiIiFSjBEl0rLE9MhxLwh+J53NZOTPiDQIAGjfVabNgeiGChEaKlP5pP6xmw+LohmzrrLcjGE1gYDaQk+cvJCzaiIiIiEg1vZN+VFtNqCgzrfh4sc1qUwZr12+maLObkZTAbEAb34v+dHKkMiw92/Y2XDthJCzaiIiIiEg1vVMLq4aQAIDTmhq2XTRFW3qwdl25ZZ07l1scsK2RMJL+6dwkRyraXVaYDLprIoyERRsRERERqUJKib4p/6rn2YAlnbYiCSMZ9YTgsplhNug3/LEuu3YK2HAsgbH5UE5mtCmMeh1219rYaSMiIiIiypXphQh84fiqyZFA8W2PHJoLoqmydFMfq3TaJn2FH0by5Wf7ICWwv9GR08/T2eDAudH5ogmqWQ2LNiIiIiJSRW86hKTduXrRVl5ihEEniqJok1Kie9yHXbWrdxbXohSwhR77/+jpUXzpmT7ce7gRr9npyunn2lvvgC8cx0j6rGCxMqi9ACIiIiK6WiASx9nReXS5vWh3WXHX7hq1l5QTStx/+xqdNp1OoNpaHAO2x+bDWAjHsavOvqmPNxv0KC81FnTs/2m3F3/ygzM40lKJv75nX85CSBSd9anv5bnR+U13MLWARRsRERFRAXj09Che6JtBl3sevVMLUFLdG8pLirdom/LDUWKE07p2JLzTVhwDtrvTgRl76jbXaQOAGpulYINIxudDeN83j6PGbsa/vPsQTIbcb+rbWWuDXidwbmwer99Xl/PPpxYWbUREREQq65/248MPnUZ5qRHXNZXj9ftqcaCpHC/0zuCrzw8gHEvAYtx4cEWh653yo8NlXbcb47SZNXGOaz3d46mibWft5jptQCr2f7IAu47BaBzv++ZxhKIJfOf3bkTVOoV4tliMenS4rDg3WtwJkizaiIiIiFR2atgLAPj++2/GziXnnfzh1EDpgZkAdm9yS10h65vy4+7O9buITqu5KBICL04soLmyFFbz5l+C19gt6JmczuKqti6ZlPjow124MObD1+6/ATtqNt9J3Iy9DQ78/NIUpJQ5346pFgaREBEREanstNuLMpN+2bwyJS59YCagxrJyatYfwVwgivY14v4VTpsZs4EoEkltJwR2j/uwewtbIwGgzVmGSV8EC+FYlla1dY+fHcePz03gE2/Yjdfsym3wyEo66+2Y8UeL4tzjali0EREREamsa8SLfY0O6HVXdwmUoq1/2q/GsnKqL50c2bHGYG2F02ZGIinhCUZzvaycCUbjGJgNYNcWtkYCWJxpp3z/CsFLl2dgtxjw3ltbVfn8ben00cHZoCqfPx9YtBERERGpKBxLoHvch+uaKpY9VmY2oNZuQX8RdtqUuP+1ZrQpimFWW8+kH1Jiy9tclSK3t4CKthNDHhzaVgGdTp2tic3p1MjhORZtRERERJQDF8Z9iCUkrmtaeQhxm7MM/dPFV7T1TflhTRel6ymGok0JIdmzxaKtqbIUJoOuYDpt86EYeqf8uL55+S8d8qWhvAQ6waKNiIiIiHLkdDqEZKVOG5DaItk/7YeU2j7P9Uq9UwtozyA5EsDiSAAtF20Xx30oM+nRWFGypefR6wS2O62LM+7UdtrthZTAoW3qFW0mgw51jhK4WbQRERERUS50jXhRYzej1rFyx6m1ugy+cBxzAe2e51pJ76Q/o/NswJJOm4ZntXWPL2BXnT0rWwg7XNaC2R55YsgDnQAONJWruo7mylJ22oiIiIgoN067vbhujRe829MhC8WUIDkfjGFqIbIsLXM1ZWYDSk16zXbapJTonth6cqSiw2XFiCeEYDSelefbilPDHuyqtW9pjEE2sGgjIiIiopzwBKIYmg2uujUSWJogWTxFW990amtfJiEkCqfNrNmibdQbwkI4vuXkSIVS7F6eUvdnIpGUODXsxfUqbo1UNFeVYnohglA0ofZScoJFGxEREZFKTo+kzrMdWCWEBAAaK0pg1IuiSpDsnVTi/jPvPDmt2i3ausdTRWq2BqQrxa5S/KqlZ3IB/kgch7apuzUSSAW0AIDbU5zdNhZtRERERCrpcnshBLC/cfUXvQa9Ds2VpUU1q613yg+LUYeG8sxDOZw2s2bPtCnJkbtqs7M9cltVGQw6sVj8quXEkAcAcH1zparrAK7E/g8V6aw2Fm1EREREKjnt9qLDZV33PFCb01pUZ9r6pvxod1k3FMqh5e2RFyd82FZVirIsnfsy6nVorS5TPYzk5LAH1VYzmiq3loiZDcU+q41FGxEREZEKpJToWieERNFWXYah2SASyeKI/e+b8m9oaySQ2h45H4ohEtfemaXu8QXsztJ5NkVHjVX1WW0nhzw41Fye0diGXKsoNcJqNhRt7D+LNiIiIiIVDM8F4QnGMopKb3OWIZpIYtQTysPKcssfiWPUG8o4OVKhxP7P+LU1+iAYjWNwNoBdWUqOVLS7bBiaDSAcU6eInfFHMDgbLIgQEgAQQhR1giSLNiIiIiIVnHYrQ7XXL9paq1MFTv+M9s+1XZ5SQkg2V7RpbYvkpYkFSJm9EBJFh8uKpFRvFMRJ5TxbgRRtQHHH/rNoIyIiIlLBabcXFqMOO2vW78C0OYsn9l85h9WRwde9lFaLNiU5ck+2i7Z0gqRa59pODnth1AvsbVg9+TTfmqtK4Z4LIlkk24iXYtFGREREpILTbi/2NThg0K//cqyqzASbxVAUYSR9U34Y9QJNFRsLr9Bu0eaD1WzYUFJmJlqry6ATQN+kOrH/J4c86Kx3wGLUq/L5V9JUWYpIPKnZlNG1sGgjIiIiyrNoPInzYz4cWCPqfykhBNqc1qLYHumeC6KpojSjYnWpqjJtFm0XJ3zYVWvbUFJmJswGPVqq1EmQjMaT6BopjKHaS2WSIPnwMTe+c3QoX0vKGhZtRERERHl2ccKHaDyJ65ozH0rcVl2GgSLYHun2BBcHIW+EyaBDRakR0/5wDlaVG1JKXBxfyPp5NkW7y6pK0dY97kMknizYom2tWW3f/NUgnjg7nq8lZQ2LNiIiIqI869pACImitboMY/NhBKPxXC0rL4bngpue66W1WW0jnhAWIvGsJ0cq2l1WDM4EEEskc/L8q1GGah9qLqyiraG8BEKs3mkLRRPoHl/AwabCWncmWLQRERER5dkptxfVVtOGzjkpYSSDM9pNx/OFY/AGY4sdkY3SWtHWPe4DkP3kSEVHjRXxpMTQbH47sCeGPWgoL0Gtw5LXz7sek0GHekfJqrPazo7OI5GUOLiBDnehYNFGRERElGfKUO2NDCVurU4nSGr4XJvyYnrTRZvVrKmQie7xBQiBjBJCN0MZUN47md+fiZNDHhwqsK2RirVi/0+7Ux3CjXS4CwWLNiIiIqI8mg/FcHk6kHEIiUIp2rR8rk0p2horttZpk1Ibke4XJ3zYVlmKMrMhJ8+/3WmFEPmN/R/zhjA+H8ahAu1WrVW0nRr2ormyFFVWc55XtXUs2oiIiIjy6OzIPABsKIQEAEpNBtQ5LOjXcOy/8mK6uWrzRVs4loQ/oo1zfd3jvpxtjQSAEpMejRUleS3aTg4X3lDtpZqrSjG9EEEomlj22Klhrya3RgIs2oiIiIjyqncqNVdrV+3GX8y3Ocs0XbS550IoLzXCbjFu6uO1NKstEIljaC64qb/njehw2dCbx1ltJ4Y8sBh1OS1Gt0JJJnV7ru62jc+HMOEL46AGt0YCLNqIiIiI8mpoNgir2YBqq2nDH9taXYb+ab9mtge+0nB6RttmOa2p4AstFG3PXJyClMDhltx2pDpcVvTPBBDPU4LkscE57G8sh3GDc/byZXFW2yti/08NpxJbDxZY4mWmCvO7TURERFSkBmYC2FZVuqEQEkVbtRUL4ThmA9EcrCz33HPBTYeQAEs6bRoII3n4uBsN5SW4qa0qp5+n3WVFNJ6E2xPa9HO80DeDN3/peYx41k4mHfOGcG7Uh1fvdG76c+Xa4qy2uVcWbR6YDIXbIVwPizYiIiKiPBqaDaClqmxTH9uajv0f0OAWyWRSYsQT2tRgbYVWtkeOekN4vm8Gb7++EXrdxovzjeioURIkN7dF0j0XxAe/exJnR+fx8DH3mvc+dWESAHB3Z+2mPlc+VJQaYTUblsX+nxr2Yl+DAyaDNssfba6aiIiIis53jg7h6e5JtZeRU7FEEiOeELZtMohje7UVANA/rb3Y/8mFMKKJ5KYHawNAeYkRBp0o+KLthydGICXwjusbc/652l2pn4m+TfxMhGMJ/P63TyCRkOist+NHp0fX3Hr7k/MTaHdZsd1p3fR6c00IgaZXJEhG40mcHZ3XZNS/gkUbERERqc4fieOT/3UOD3zjOL78bJ9mz2ytZ8wbQjwp0VK9uU5bQ0UJTHqdJsNIlDNGW9keqdMJVFsLe8B2MinxyIkR3NxWtaWuYqasZgPqHRb0bXBWm5QSn/jRWZwf8+EL77wOD9zWCvdcCMeHPCve7wlEcXRgDnd31mRj2TnVXFlyVdF2ccKHSDyp2eRIgEUbERERFYDTw14kJbCvwYG/+8klfPThLkTiyyO7tW4wXbhsdnukXiewraoU/Rqc1Ta8xcHaCqetsAdsHx2Yw/BcEPfekPsum2K7y7rh2P9v/2oIPzw5ig/f1YG7dtfg7s5alBj1+OHJkRXvf/riFBJJWdBbIxXbqsrgngsimUz98ue0W9shJACLNiIiIioAxwbnoBPAd993Iz7yazvww1OjePdXj2JOo4EbqxmaTRVbLZvcHgmkEiS1eKbN7QlBJ4D68s1vjwSuDNguVI+ccMNmNuB1nXV5+5wdLhv6pvyLRcp6TgzN4a8ev4DX7HTiw3d1AADKzAa8bm8tHj8zjnBs+S9MfnJ+AvUOC/Y1OLK69lxoqixFJJ5cLO5PDXvhsplR77CovLLNY9FGREREqjs+NIfddXbYLEZ86K4OfOldB3FmZB73fPkF9E3lbwZVrg3MBFBi1C8GamxGq7MMQ7P5i3jPFvdcEHWOki1HxTtV3h756OlRPNY1tuJjC+EYnjg7jjcdqEeJSZ+3Ne2osSIUSyybTbaSqYUw/uDbJ1FfXoIv/NZB6JYEpbztUAMWwnE8c3Hqqo8JRuP4Zc80fr2zdlOpp/m2GPuf7u6eGvbgYHO5Jta+moz/1Qgh9EKIU0KIx9PvtwohjgoheoUQ3xdCmNLXzen3+9KPt+Rm6URERFQMYokkTg17cUNL5eK1Nx+ox0PvvwnBaALv/MqvENNYgVbpikwAACAASURBVLKaodngpuP+FdurrYglJEa9m494V8PwFuP+FU6bGbOBKBIZdpWy7Qs/68WHvncKPzixfBthqkuVxL2H87c1EgD2prtfZ0bm1733U493YyEcx7+++3o4Sq8ecn7L9mrU2M344cnRq67/smcakXgSv66B82zA1bPa5gJRDM4GNb01EthYp+3DALqXvP8ZAJ+XUnYA8AB4IH39AQAeKWU7gM+n7yMiIiJa0YUxH4LRxLIhxAebK/Bnb9yFGX9Uk2e4VjI4G0DrJkNIFG3p2P/LGkuQdM8Ft5QcqagrtyCRlBhToWhNJCVGPEGY9Dr86Q+68OOz41c9/vBxNzpc1rynFO6stcFs0KErfXZrNVJKvHR5Fq/fV7vivDK9TuAt1zXg55emMLvk3OBPzk+ivNSII0t+sVLIGspLIERqVttpdypY5aCGkyOBDIs2IUQjgDcC+Gr6fQHgTgA/SN/yDQD3pN9+S/p9pB+/S2i5F0lEREQ5dWxwDgCu6rQpOutTHYQL4+t3EApdIinhngti2yZDSBSLEe8bDJ5QUyiawNRCJCudNuVM1dnR/P9MjM+HEEtIfPz1u3CwuQIfeugUfn4ptZWwb2oBp4a9uPdwU9634Rn1OuxtcKBrZO2ibdQbwow/smYB89aDDYgnJR4/kypIY4kknu6exF27amDY4tbWfDEZdKh3lMA9F8SpYS/0OoF9jYV/Fm8tmX7nvwDgTwEoexOqAHillPH0+yMAGtJvNwBwA0D68fn0/URERETLHB/0oLmyFDX25SEBbdVlMBl0uDDmU2Fl2TXmTb3g30oICQCUl5pQbTVpqmgbSZ+1ykYE/s5aG4x6kdFWwGxTzkjtqrXhwd+9ATtqbPjAt07gaP8sHjk+Ar1O4J6DDes8S24caCzH2dH5Nc86drlT37MDaxRtu+vs2FVrww9PpbZI/qp/Fr5wXBNR/0s1pWP/Tw17sbPGhlKTQe0lbcm6RZsQ4k0ApqSUJ5ZeXuFWmcFjS5/3/UKI40KI49PT0xktloiIiIqLlBLHh+aWbY1UGPQ67Kq14cK49ou2oXTc/1Y7bQCw3WnVVNHmzmLRZjbosavWjjPrdJVywT135etwlBjxzfceQWNFCR74xnF8/7gbd+5ybSlkZisONDkQjiXRs8a8ttNuD0wGHXbVLt8audTbDjWgy+3F5Wk/fnJ+AiVGPe7Y4cz2knOqubIUQ7MBdLm9mp7Ppsik03YrgN8QQgwCeAipbZFfAFAuhFBK1kYASozOCIAmAEg/7gAw98onlVJ+RUp5WEp52OnU1g8BERERZcfgbBAz/uiKWyMVnfV2nB/zaX7g9oAS91+99cKlPT2XSyvfk2wM1l5qX6MDZ0fnM464z5ah2SAMOoG6dHR8ldWM7/zeTagoM8IbjOHew015Xc9SBxpThclaxWyXex6d9XaYDGuXAG+5rgE6Afzw5Ah+en4Sr9rhhMWYvzTMbNhWVYYZfxQLkbjmQ0iADIo2KeX/lFI2SilbALwTwDNSyv8B4FkAv5m+7X4Aj6bffiz9PtKPPyO18n8UIiIiyqsr59lWf1G1p84ObzCG8flwvpaVE0MzAViMOtTYtj4rqt1lxUI4XtDzypYanguh1KRHVZkpK8+3v8GBhXAcQ3PrR9xn0/BcEA0VJVed7ap1WPC9992Ev3jzHrxmp3qNiG1Vqe7faufa4okkzo7OZxSSUmO34Nb2ajz4/CCmFiK4e6+2tkYCV3d1r5VO22o+BuAjQog+pM6sfS19/WsAqtLXPwLg41tbIhERERWrYwNzqCg1YrvTuuo9e+pTW7m0fq5tcDaIbZVlV83F2qwOlw2AdsJI3J4gmiq2NupgKSVUIt9bJN2rjC1orCjF797aqmpQhxACB5rKcdq98lm/nkk/QrFExsmWbzvUgFAsAYNO4M6d2ivalL8nR4kRrVnYkqy2Df1kSSl/LqV8U/rtfinlESllu5TyHVLKSPp6OP1+e/rx/lwsnIiIiLTv+JAHh1sq13wxv7PWDiGg+XNtQ7MBbNtiCIliMUFSI7H/qbj/7HztALCjJhVxfzbPYSTZmjWXK9c1OtAzuYBgNL7ssdPpcQCZFm13d9ai1KTHzdurls1z0wLl7+m6pvKs/KJEbdrI7SQiIqKiM70QwcBMYM2tkQBgNRvQUlWm6U5bMikxNBdEyxZntClq7GZYzQZNdNqklFkvdox6HfbU23Emj7H/vnAMnmCsoIu2A03lSCQlzq/wb+W024OKUmPG6y81GfDN9x7BX9+zN9vLzIuKUiP2NThwd2et2kvJCm1nXxIREZFmnRhKnWc7nMHA3j11dpwZzX9aYLZM+MKIxpNZ67QJIbDdpY0EyblAFMFoIiuDtZfa3+DAD06MIJGU0Oehk5LtMJVc2J8OI+lye5eF+3S553GgqXxDW1Qz+bdZqIQQ+O8/vE3tZWQNO21ERESkimODHpgNOuytX3/o7Z56O9xzIcyHYnlYWfYNzqSTI7N4tqbdmUqQLHTKbLNsFzv7GssRiCYwMJOf74ES99+cpcI7F5w2MxrKSxa3Qir8kTh6phYWEyZJe1i0ERERkSqOD87huqbydePHgSthJBc1eq5tMN2lydb2SCB1rm16IVLwhWyuirb9i2Ek+dkiOTyXvVlzuXSgybEsQfLsyDykBK4rghTFaxWLNiIiIsq7YDSOc2O+NeezLdVZl06Q1GjRNjQbgMmgQ51963H/isUwkgLvto14QgBSCYvZtN1pRYlRn9eiraLUCLulsEM5DjSWwz0XwlwgunhtMYSEnTbNYtFGREREeXdq2ItEUuKG1syKNqfNjGqrSbNhJIOzATRXlmY1xa4jXbRdLvCibXg2CKfNjBJTdocz63UCexvsOJunMJJCT45UHEinQy7ttnW5vdhWVYqKLM3Jo/xj0UZERER5d2xwDjoBHMpwu5YQArvr7BrutAXRkuWzUE2VpTAZdAUf+5/LYmd/YznOj80jnkjm5PmXGs7y2IJc2dfggE6kCjXFabc346h/Kkws2oiIiCjvjg96sKvWDtsGtpp11qdmUEXjuX+Bnk1SSgzOBrAtywN+9TqBtuqygt8emRqsnd3kSMX+RgfCsWTOA1niiSRGPaGspX/mUpnZgA6XbbFom5gPY8IXZgiJxrFoIyIioryKJ5I4OexZdz7bK+2ptyOWkAVfpLzSpC+CcCyZ1RASRaHH/scSSYx5QznrtO1rSIWR5HrI9vh8GPGk1MT2SEAJI5mHlPLKeTaGkGgaizYiIiLKq55JP4LRBA5t22DRptEwksFZJe4/+y/4251WuD1BhGOJrD93Nox5Q0jK3CUutlSVwWY25HyGn1aSIxX7G8sxF4hixBPCabcXRr1Y/PdD2sSijYiIiPKqZ3IBALB7gy8iW6vLYDHqNBdGMjSb/RltinaXFVIClwv0XJt7LpUcmatiR6cT2NvgyHmnLVdjC3JFOb922u1Fl9uL3XV2WIzZDYKh/GLRRkRERHnVM7kAo15suIjR6wR21dpxYTw/aYHZMjgbhFEvUOfIXty/oqOmsGP/81Hs7G90oHs8t2cdh+eUv8PcnM3Ltp21NpgMOpwa9uLMiJfn2YoAizYiIiLKq57JBbRWl2U0VPuVOuvtuDDmg5QyByvLjaHZAJoqSmHQZ/9lV2t1GXSicGP/h+eCMOl1qMnifLpX2tfoQDSRXOzg5sLwXBCNFaXQZ3FkQy4Z9Trsrbfjsa4xBKIJJkcWARZtRERElFc9k3501Ng29bF76u3wheOLA5u1YHAmmJMQEgAwG/Roriwt2Nh/tyeIhoqSnBY7Shcpl0O23RqJ+1/qQFM5ZvyRxbdJ21i0ERERFYiHj7txctij9jJyKhRNwO0JYodrk0WbxsJIrsT95+4Ff3sBJ0jmo9hprChBeakRZ3MYRjI0G0RzpTa2RiqU7prNYkBbjn5pQPnDoo2IiKgAxBJJ/PmPzuEPv3uqYJMAs6Fvyg8pgZ211k19/K5aO3QCmgkjmfZHEIwmchJCotjusmJgJpCXAdNAarvnC30z697nj8TRPx3IebEjhMC+BkfOOm3zwRjmQzHNhJAolA7kgcZy6DSyrZNWx6KNiIioAFye9iOaSGLUG8I3XhxUezk5cyl97miz2yNLTHq0VpdpptM2NJsK4shpp81pRSwhMZQO/ci1zz55Cfc9+PLi8ObVfP6pHgSicbz9UGPO17S/0YFLEwtb+oXH2ZF5xFYofN0ebSVHKrZVlWJnjQ137XapvRTKAhZtREREBUDpHHW4rPinZ/swF4iqvKLc6J1cgEmvw7YtvADurHdoptM2OJO7uH+FUgDna4vk2dF5JJISH32ka9Ui6cKYD19/cRDvOtKMg80bm8e3GfsayhFPSnRvspg/OzKPN//T8/jCz3qWPXYlAVNbWwyFEPjJH92B99zaqvZSKAtYtBERERWAC2M+mA06fPFdBxGIxPHFp3vVXlJO9EwuoM1ZtqUkxT31dox6Q/AGC7+wHZoNwqATaKzI3RbB7c5UMZGPom0+FMPwXBC3tlehb8qPzz+1vMhJJiU++eg5lJcY8ad378z5moBUgiQAnNtkMf+tXw0CAB58fhBTC+GrHlO6pU0aO9NGxYVFGxERUQG4MO7DzlobdtfZ8c4jzfj2r4bQX6CJgFvRM+nHztrNbY1UqB1G8uylKdz8t0/jFz3T6947MBNAY0VJTuL+FTaLEbV2S15i/5UO5/vv2I53HWnCV57rx4mhuavueeSEGyeGPPifb9iN8lJTztcEAPUOCypKjTg/uvFzbfPBGB7rGsPtHdWIJpL452cvX/X48FwQlWUm2CzGbC2XaMNYtBEREalMytS2LqUY+aPX7oDZoMP/+fFFlVeWXf5IHKPeEHZs8jybYrdStKm0RfK5nhmMz4fx3q8fw7deGlzxnkRS4nM/vYQnzo3jUB62B7a7rHmJ/T8/liqKOuvt+LM37kG9owR//MgZhKKpbZJzgSj+9scXcaSlEm8/1JDz9SiEEOisd+D8Jn4m/vPkCMKxJD72ul2493AjvnN0CO4l5wO1GPdPxYdFGxERkcomfGF4gjHsqU8VI06bGX/w6u346YVJHO2fVXl12dOrhJC4NpccqXDazHDZzKp12i5P+9HusuI1O5345KPn8RePnb8quXF6IYLf+dpRfOmZPtx7fRM+/bZ9OV+TEvufTOZ26Pi50XnU2i2otpphNRvwd+/Yj4GZAD7zZOoXDJ998iL84Tg+dc9eCJHfxMLOBjsuTSysGCayGiklvn10CAeby7G3wYEP3dUBIQS+8LMr25OH54JbOoNJlA0s2oiIiFSmdIyUThsAPHBbG2rtFvzNE905fyGeL72TqU7QVrdHAqlOj1qdtsvTfuyps+Pffucw3nd7K77+4iB+75vHsRCO4eWBObzxi8/hxJAHn/3N/fjMb+6HxajP+Zq2u6wIRhMY94XXv3kLzo35sLfhys/pLdur8bu3tODrLw7in3/eh4eOufHAba1Z+TveqM56B6KJ5OLPWSZeujyL/ukA3n3jNgBAnaME99+8DT86NYLeyQXE04muWkuOpOLDoo2IiEhlSvGxa0nRVmLS40/u3okzI/N4rGtMraVl1aXJBViMOjRVbP0F8J56O/qm/IjE8zvTLhxLYNQbwnanFXqdwJ+9cQ/+9m378HzvDF7/j8/hXf/+K5Sa9PivD96Kew835W1dSvcyl2EkwWgc/dN+dNY7rrr+p6/biZaqUnz2yUuod1jwobs6craGtexNd6rPjWV+ru3bR4dQXmrEG/fXLV77g1e3o9RkwOd+2oMxbxiJpGTRRqpj0UZERKSyC+M+tFSVwmo2XHX9rQcb0Flvx9/95BISRdBt65lcQLvLmpVBv3vqHIgn5Ya6KtkwMBOAlECb80r8+7uONOMb7z2CQCSO13XW4rE/vG3x3F2+tKeLNmULai50jy8gKVNdzqVKTQZ87t4DqCoz4VP37EXZK36O86WlqgxlJn3GHdhJXxg/OT+Jew83XdUNrSwz4fdub8WT5yfw32dSvzDhmTZSG4s2IiIilV0Y9634Il+nE3jf7W0Y9YZwbhOpeIWmd9KPHa7sbJtTzv/le4vk5XTYx3bn1efybm2vxvE//zV8+X8cgl2FlMGqMhNq7GacGcndz4kSQrK3wbHsseu3VeLYn70Wd+2uydnnX49OJ7C7zp7xv5WHXnYjkZT47SPNyx77vdvbUFlmwj+mR28053A4OlEmWLQRERGpaCEcw9Bs8KrzbEvd1lENAHiud/14+UI2H4phwhfGjiydddpWWZrqquQ5jOTyVABCAK3Vywct67PQQdwsIQSu31aBk8OenH2O86M+VJaZUOewrPh4NjqoW7W3wYEL4751z4HGE0l87+Vh3LHDiZYV/i6tZgP+n1dvRzSehEmvQ6195a+ZKF9YtBEREano0kRqO9ue+pWLtmqrGXvq7Hiudyafy8o6ZdvejpqtJUcqlK6KGp22hvISlJhyHy6yUYeaKzDiCWEyR2Ek58bm0Vlvz3sq5EbsqbcjGE1gYDaw5n0/657ChC+Md9+4vMumePdN21DnsKCxskTVgpwIYNFGRESkKqVTtFrRBgC3d1Tj5LAHgUg8X8vKup702bOOLG2PBFLfs0y6KtnUP+NftjWyUBzalpoHd3JoY922cCyBH5wYwW/920v4wYmRFe+JxBPomVxYFkJSaPam17fevLbvHB1CvcOCO3e5Vr3HYtTj3+87jM+8fX9W10i0GSzaiIiIVHRhzIeKUuOa269u73AilpB4eWAujyvLrp7JBZSa9GgoL8nac+6ps8MficPtCa5/cxYkkxKXpwJXhZAUks56O0wGHU5kWLSNeIL4zJMXcfPfPo0/fqQLJ4c9+NxPL60456x30o9YQl4V91+IOmqsMOl1OL/GubaBmQCe653Bu440w6Bf+6Xw3gYHbmipzPYyiTZMnXgfIiIiApDqtO1ZZ8vZ4ZYKmA06/LJ3Gq9ZozNQyHomF9BRY8vquaelYSTbqnJfSE34wgjFEgXbaTMb9Njf4Fj3XJs3GMXH/vMMnrowCQD4tT01uP/mFgSiCbzvm8fx0/OTV0XgA0tCSAq802bU67Cz1rZmp+2hY8Mw6AR+60j+RjIQbRU7bURERCqJJ5K4OLGwagiJwmLU40hrJZ7X8Lm2nkk/driyW+zsqLFBrxN5CyNZLTmykBzaVoFzoz6EY6vPr/vO0WH85PwkPvCq7XjuY3fi337nMG5pr8adu1xorizF118cWPYx50Z9sJkNmphX1llvx7mxeUi5fNuslBKPd43j9o5quGwMFyHtYNFGRESkkv6ZAKLxZEYzvW5rr0bvlB8T87kJmcglTyCKGX8EO2qyd54NSBWz7U5r3sJILqcHV293Feb2SCAVRhJNJBc7Yyt5unsS+xsd+Njrdl21XVWvE7j/lhYcG/Qsi80/NzaP3fX2gkiIXE9ngwPeYAxjK/xbOeX2YtQbwpv216uwMqLNY9FGRESkEqXYWCuERHF7hxMA8Hyf9rptPenkyI4sJUcupYSR5EP/TAA2iwFOqzkvn28zDm0rBwCcHPKu+PisP4JTbu+qARzvONyIUpMe//HC4OK1RFKie9xX8FsjFcrw75XmtT3eNQ6TXodf61RvnhzRZrBoIyIiUkn3uA8mvS6j7Xa7am2otpo0Oa9NKdp2ZmlG21J76uwYnw9jLhDN+nO/0uXpVHJkIUfeu2wWNFeWrhpG8oueaUiJVYs2u8WI37y+Ef/dNYYZfwQA0D/tRziWLPgQEsXuWjt0YnmCZDIp8cTZcbxqp1OVAehEW8GijYiISCUXxn3YUWuFcZ0EOyA1l+zW9mq80DeT14j7bOiZ9MNmNuRkQPHSMJJcK+TkyKUONZfjxLBnxTNdT1+cgtNmXrNrdt/NLYgmkvje0WEAqa2RQCpJUQtKTHpsd1qXJUieGPZgwhfGm14RskKkBSzaiIiIVCClxIUx37ohJEvd1l6NGX8UF9MDubUilRyZmw6V8v27ML76Ga5s8EfimPCFCzqERHH9tgpML0Qw4glddT2WSOKXl6Zx507XmmfT2l1W3LHDiW/9agjReBLnRn2wGHVoqy78glWxt8GxrNP2eNcYzAYd7trNrZGkPSzaiIiIVDC1EMFsILqhou3KuTbtbJGUUqJnciEnWyMBoKLMhHqHJeedtn4NJEcqFodsvyL6//igBwuReEZjI95zSwumFiL48blxnBudx65a+7ozzQpJZ70dE77w4hbPRFLiiXMTuHOXC1YzJ16R9mjnXx8REVERuRJCkvmWs1qHBR0uK57TUPT/jD8KTzCGDlduijYgP2Ek/dMBAEB7ASdHKnbW2FBq0i871/bMxUmY9Drc1lG97nO8aocTrdVlePCFQVwY92nmPJtC2TardNuODsxieiHC1EjSLBZtREREKlCKjF11GytmbuuoxssDc2vO4SokvekQkmzH/S+1p86Oy9OBnH5PLk/7odcJNFcWftFm0OtwXVP5sqLt6YtTuLGtMqNOk04ncP/N29Dl9mIhHNdMcqSiM71eJUHy8TPjKDHqVw1gISp0LNqIiIhUcGHMh6bKkg2n2N3R4UQknsTxwZXTAQtNz2LRlrtthXvq7UgkJS7l8Kzf5Wk/mitLYTJo46XT9dsq0D3uQyASBwAMzgTQPx3YUNHy9usbFws8rYSQKBwlRjRVluDCmA/xRBJPnpvAXbtdKDHp1V4a0aZo4/88RERERaZ7fGMhJIob2yph1As8p+K5tvlQDB//zzN49PQoovHkiveEYwn888/78LmnelDnsMBpy91ssz11qYIil1skL08FsF0DyZGKQ9sqkJRA10hqXtszF6cArB71vxKbxYh33tAEm9mQkxl7uba33oFzY/N48fIs5gJRbo0kTWPRRkRElGeBSBwDs4HFYmMjSk0GHGquwHM96p1re7p7Eg8dc+PDD53GrZ95Bp9/qgeTvjCAVODDw8fdeM3f/xyfffISjrRU4lsP3JjT2WaNFSWwmQ1bDiN59uIUftmzvBhOJCUGZgOaCCFRHGpKh5Gkt0g+e2kK251l2Fa1scLzY6/fhac+8iqYDdrrUHXW2zE0G8RDx4ZhNRvw6p1OtZdEtGmMzyEiIsqzC+M+SJl6UbkZt3dU4+9/2oMZfwTV1tx1sFZzctgDq9mAL77rOnzrpSH849O9+PKzfbi7sxZ9U35cmlzAgUYHPv9b1+Gmtqqcr0enE9i9xTCSb700iE8+eh5mgw5PfPj2qwq0UU8I0XhSU0Wbo9SIdpcVJ4Y88Efi+FX/LN5za+uGn8eo16HWkf35evnQmd7S+cTZCbz1YAMsRu0VnkQKdtqIiIjyrMud2rK2v2lz54SU6P8X+tTptp0c8uK6pnLcuasG//GeI/j5H78av3tLC57rnUYolsA//fZB/NcHb81LwabYU2dH97hvU4PH//UXl/HJR8/j1TudKDHp8dGHuxBPXNn2eVmJ+9dAcuRS1zdX4JTbi+d6phFLyGsuhGPpL0U4UJu0jkUbERFRnp12e9FQXgKXbXMdjL0NDtjMBhwdmMvyytYXjMZxccKHQ83li9daqsvw52/ag1P/69fxiz95Nd60vz6n2yFXsqfejmA0gaG5YMYfI6XE5356Cf/nxxfx5gP1+Pf7DuOv3rIXp91e/Nsv+xfvU4q2tmrtdNoA4NC2cniDMXz1+QHYLAZcn57fdq1w2Sxw2cywWQyLv+gg0ipujyQiIsqzMyPz2N+4+TQ+vU7gcEsFXlahaOtyzyMpgYPNywsAvS6/hdpSSqjL+bF5tFav3xGTUuJTj3fjwRcG8M4bmvA3b90HvU7gzfvr8OS5cXzhZz24a7cLu2rtuDztR2WZCRVlplx/GVmlFGknhjx40/46GDU0HDtbHritFUa9TjOpn0Sr4U8wERFRHs0FohieC+JAU/n6N6/hSGsV+qb8mPFHsrSyzJwcTgVbHGze2vqzraPGCoNOZBRGIqXEJ350Fg++MID33tqKv33bvsWCUwiBT71lLxwlRnz04S7EEknNJUcq2qqtcJSkRkrctfva2hqp+MCrtuO9t238LB9RoVm3aBNCWIQQLwshuoQQ54UQf5m+3iqEOCqE6BVCfF8IYUpfN6ff70s/3pLbL4GIiEg7lAj2A41bLdoqAQDHB/PbbTs17EGbswzlpYXVdTIb9NhZa8OpYe+6954Y8uB7L7vxgTva8Mk37V62lbPKasbfvHUfzo/58E/P9KF/xq+pEBKFTidwqLkcQgCv2nFtFm1ExSKTTlsEwJ1SygMArgPwOiHETQA+A+DzUsoOAB4AD6TvfwCAR0rZDuDz6fuIiIjWFU8kIeXGgyS0pMvthRDAvi1sjwSAfQ0OWIw6vDyQvyHbUkqcGvbiYFNhno26ZXsVTgx5EIom1rzv+b4ZCAH8wau3r3r27u7OWrz1YAP+6dk+zPijmizaAOD3X7Udn3zjHlRqbGsnEV1t3aJNpvjT7xrT/0kAdwL4Qfr6NwDck377Len3kX78LpHv08hERKRJf/xIF177D7/AiCfzMAmt6XJ70eGywmre2rFyk0GHg00VeHlwNksrW9/wXBCzgSgObSusrZGK2zuciCaSODqw9vfkxb5Z7K13rNst/Is3d6LamrpHa8mRihvbqrg9kKgIZHSmTQihF0KcBjAF4CkAlwF4pZTx9C0jABrSbzcAcANA+vF5APnL/CUiIk1KJCV+1j2Fy9MB3PuvL2FgJqD2krJOSomukfktb41UHGmtxIUxHxbCsaw833qU82yHVgghKQRHWithMujwfO/qoxCC0ThOuT24pX39lyaOUiP+/h0H0Fpdhv1Z+jsjItqMjIo2KWVCSnkdgEYARwDsXum29J8rddWW7XURQrxfCHFcCHF8eno60/USEVGR6h73wR+J4/dftR2ReBLv+NeXI8F/hwAAIABJREFUcGliQe1lZdWIJ4S5QHTLISSKG1srkZSpM1r5cGrYizKTHjtqbHn5fBtlMepxQ0sFnl9jft3LA3OIJSRu3V6d0XPe3uHEs3/8alWGmBMRKTaUHiml9AL4OYCbAJQLIZS9HY0AxtJvjwBoAoD04w4Ay05JSym/IqU8LKU87HRydgYR0bVOmTl2/y3b8P0P3Ay9Dvitr7yEMyPrB0toxen0UO3rslS0HWyugEEn8hb9f3LYgwNN5apG+6/ntnYnLk4sYMoXXvHxFy/PwqTX4YaWyjyvjIho8zJJj3QKIcrTb5cAeC2AbgDPAvjN9G33A3g0/fZj6feRfvwZWeynyomIaMteHphFU2UJ6hwlaHdZ8cgHboHNYsBv//tRVeaR5UKX2wuTQYedtdnpVJWY9NjX6MjL9ycYjaN7fKFgt0Yqbu9IddBW67a90DeDg83lKDHp87ksIqItyaTTVgfgWSHEGQDHADwlpXwcwMcAfEQI0YfUmbWvpe//GoCq9PWPAPh49pdNRETFREqJY4MeHGm5cs6ouaoUD3/gZrjsZtz34FGMekMqrjA7zozMo7PentUhx0daK9E14kU4tnZi4ladHZlHIikLbj7bK+2ps6OyzLTiuTZPIIoL4z7c2p7Z1kgiokKRSXrkGSnlQSnlfinlXinlX6Wv90spj0gp26WU75BSRtLXw+n329OP9+f6iyAiIm27PO3HXCCKG1uv3rJW5yjBV+87jHAsiZ9dmFRpddkRTyRxdjR7ISSKG1srEUvIxa2XuXIyPf/sYIF32nQ6gVvbq/F838yy8REv9c9CSuDWDEJIiIgKSfZ+1UdERLRJynm2G1qXnzNqc1qxraoUv+zRdmhV75QfoVgia+fZFNdvq4QQyPkWyZPDHrRWl2li3tft7dWYWoigZ9J/1fUX+mZQZtIzCZKINIdFGxERqe7lgTk4bWa0VJWu+Pirdjjx4uVZROK53QKYS13pTli2kiMVjhIjdtXac1q0pYZqe3Awy2vPldvS59qe67260H/x8ixubKvK6vZUIqJ84P+1iIhIVVJKvDwwhyOtlRBi5VTCOzqcCMUSODGYn2j7XOga8cJuMaxamG7Fja2VODHkQSyRzPpzA6lRBTP+KA5uK+ytkYr68hJsd5bhuSXn2sa8IQzMBHDLdm6NJCLtYdFGRESqGvGEMD4fXnaebambt1fBqBf4Ra92t0ieds/jQFP5qoXpVhxprUQolsD5Md+mn8MTiOITPzqLT/zoLPyR+FWPXRmqrY1OG5Car3Z04Ep39oV0miRDSIhIi1i0ERGRqpRtfWvNzSozG3B4WyV+cUmbRVswGkfP5ELWz7MplO/dywOzG/5YKSX+69Qo7vqHX+DhY2489PIw3vyl53F+bH7xnpNDHpSa9NhZoEO1V3JbezXCseTi4PEXL8+iqsykqa+BiEjBoo2IiFT18sAc7BbDui+m79ix9tDkQnZ+zIdEUuYsAMNpM6OtumzD59qGZ4O478GX8f99/zSaK0vx+Iduw3ffdxOC0Tje+uUX8c2XBlPn2dxe7G90wKChs2A3ba+CQSfwXG8qRfKFvhncvL0KugIeDE5EtBrt/N+XiIiK0rHB1Hm29V5M37Ejta3tlyvM3yp0iyEkjY6cfY4jrZU4NuhBMinXvxnAV5/rx69/4Rc4NezFX72lE//5B7dgV60dN7VV4YkP3Y5b26vwvx49jw986wQujPkKfqj2K1nNBhxqrsDzvTO4PO3H1EKEWyOJSLNYtBERkWqmFsLonwngyBrn2RR76uxw2syajP7vGplHvcMCl92Ss89xpLUS86EYeqYW1r333Og8/vr/78bNbVV46iN34L6bW6BfUjRXWc342v034M/esBvPXJxCPCk1V7QBqRTJc2PzePzMOADg1u0s2ohIm1i0ERGRao4NpM4bHWldP9FPCIHbO6rxXO80Ehl2kwpFl9ub9aj/V7pyrm39LZLHBlP3fPpt+1DnKFnxHp3u/7Z33/FVV/cfx18nm0yygQRCQgIh7IAIsgXcirNOtFqLtdpWO/21Vjvs72eXba11Va17VhE3ggKCsiHMJBBGFhkkIXvnfn9/5KKMJGTcm3sD7+fjkUcu3+/5fu/n5pDkfnLO+RzDd2cm8Nb3pnLrtKF9cpRqelIElgXPrD5AbGg/hjihcqeISG9Q0iYiIi6z4UAp/bw9GTUouFPtZw2P5EhtEzvzK07d2E2U1TSSU1br9KQtNrQfg0L8WL//1EnblpxyBob4tZuwHWvCkFAevHQU/Xw8HRFmrxobE0KQnxfVDc0aZRORPk1Jm4iIm1u3v5QL/v7F11XwTicbDh5hYlxopzc7np4YgTH0qSmS2/KOrmdzbtJmjGFKQjjr9pdiWR2PRG7JPtInpzt2lZenx9f7sp2TqP3ZRKTvUtImIuLG3tyUy8Jn15NRWMXbW/JcHY5DVdQ2kVFY2an1bEeFB/oyJiaEVW6YtNlsFvsPV/N5RhGvrs/hkU8z+dlb2/i/j9IxBsY4sQjJUVOHhVNa08ieoup22xRV1pNfXseEPrTnWk9cOHogAT6efXJ6p4jIUV6uDkBERE5ms1n8aWkmT67ax/TECGyWxZo+WDWxI5uyy7AsupS0AcxMiuSJVfuorG8i2M/bSdGd2uGqBtJyy9mWW862vNbPlfXfbErtYSAqyI8BIX7cNTuRQF/n/8qdah9VWruvhBED2t5CYYt9xDY17vQfaQNYMH4Q542Kxt9Hb3lEpO/STzARETdT29jMvW+ksXRXETecPYTfXjaKV9fn8OB7u8gurSEuPMDVITrEhoNleHuaLm84PXN4JI+tyOKrrBIuGD3QSdF1bPehSi7552psFnh6GJIHBHHJuEGMj+3PsKhABvX3IzLQt9f3NYsN9WdImD9f7Svl29Pi22yzJecIPl4enV5H2NcZY5SwiUifp59iIiJupLiyntte2MjuQ5U8cEkKt04bijGG6UmtU7tW7y05fZK2A2WMi+2Pn3fXClxMGNKfIF8vVu057LKkbUVmMTYLXr39bCYMCXWrIh1TE8L5ZFchLTbruDL+R23JKWdMTAi+Xu4Ts4iIdExr2kRE3MjvPtjNvuIanrllErdNj8eY1jfdCREBDArxO22mSFbUNrEjr4KzE7o2NRLA29ODcxLD+WJPySkLbjjLuv2ljIgO4pzECLdK2KC14EZFXRPpBZUnnWtstrEjv4LUM2Q9m4jI6UJJm4iImzhUXsfHOwtZODWOc5OjjzvXukdZJF/tK6G5xeaiCB3n88wimm0W80ZGn7pxG2YOjyS/vI59h2scHNmpNTbb2HTwCFO6kXD2hqkJrevavtp3coK/61AFjc22M6JypIjI6URJm4iIm3hxbTaWZXHz1Lg2z09PiqCyvpntfWiPsvYs3VlEVJBvt8vgz0yKBFxT+n9Hfjl1TS1fF/1wN1HBfiREBrB2X+lJ5zafYUVIREROF0raRETcQG1jM69tyOGC0QOIDfVvs800+x5lfX2KZH1TC6v2HOa8UdF4tLHmqjMGh/kzLDKAFZnFDo7u1NbZN6+eHO+eSRvAOcPC2XCgjKYTRmW35pQT078f0cF+LopMRES6Q0mbiIgbWLw1n4q6Jm5tp+IfQFiAD6MGBff5pG313hLqmlo4f9SAHt1nXko06/aXUlnf5KDIOmftvlKSBwQRFuDTq8/bFVMTIqhpbGHHCaOyW3KOaJRNRKQPUtImIuJiNpvFc2sOMCYmhEmneEM9IymSLTlHqG5o7rCdO1u6q5BgPy+mJPRspGr+yGiaWqxenSLZ2GxjU3ZZj2N3tqPr7Y6dIllQUUdBRb2KkIiI9EFK2kREXGx1Vgn7Dtdw2/ShX1eLbM+MxAiabRbr95+8XqkvaG6x8Vl6EXNHRuPdwz3MJgwJJSzAh+W7ixwU3altyyunvsnm9klbeKAvyQOCjkvatmSXA6gIiYhIH6SkTUTExZ5bc4DIIF8uHjPolG0nDg3Fz9uD1X10iuSGg2UcqW3ivJTuVY08lqeH4dzkKD7PKD5p7ZazrNtXijG4beXIY00dFs7Gg2U0NLcArUVIfL08GDnwzNhUW0TkdKKkTUTEhbKKq1m15zALp8Th43XqH8m+Xp5Mjg9n9d7er5roCJ/uKsLXy4NZIyIdcr95I6OprG9m48Eyh9zvVNYdKCV5QDD9/d13PdtRUxPCaWi2kZbTOsK2JecI42L7d+r/mYiIuBf95BYRcaHnvzqAj5cHN5w9pNPXzEyKYN/hGgoq6pwYmeNZlsWy3UXMSIrE38fLIfecOTwCHy8Plu92fhXJhuYWNh088vU+aO7u7IRwPAx8ta+U+qYWdh2qYEKc1rOJiPRFStpERFykvLaRtzfns2DcICICfTt93fSkCIA+N0VyZ34l+eV1nD+q51Mjj/L38WJ6YgTL0guxLMth923LttwKGpptfWJqJEBIP29GDQph7f5Sdh2qoKnF0no2EZE+SkmbiIiLvL4xl7qmlg7L/LdlRHQQkUG+fS5pW7qrEA8Dc0c6LmmD1imSuWV17C2uduh9T7Ruf+t6trPdeH+2E50zLJytOUdYs7e1IImSNhGRvklJm4i4vaYWG79cvIPPM3qvSqCzHa5q4Okv9nPOsHBSBnWtMIQxhumJEXyZVYLN5tzRJUdauquQyfFhDt/fbO7IKACWOaCK5Cc7C7nv7e3UtLGlwtp9paQMDCbE37vHz9NbpgwLp6nF4qV1Bxkc1o/IoM6P6IqIiPtQ0iYibu/hjzN4dX0O//7igKtDcQjLsvjV4h1UNzTz28tGdese0xMjKKtpZHdBpYOjc479h6vZW1zd4w212xId7Me42JAeJ21ZxVXc+0Yar2/MZeGz66mo+2bT7vqmFrbkHHH7Uv8nOmtoGF4ehpLqRiZqlE1EpM9S0iYibm1JWj7PrjlAeIAPGw+WUVXfdOqL3Ny7afl8uruIn543nKTooG7dY4Z9XduarL4xRfJTe0J1nhOSNoD5KdGk5ZZTXFXfrevrm1r4wWtp9PPx5PeXj2ZHfgU3/HsdpdUNAKTlltPQbOszRUiOCvT1YmxsCACpp9i4XURE3JeSNhFxW+kFlfzi7e1MHhrGP66bQLPN4sss99lUem9RFV/sOcyeoioq6po6VQijsKKeB5fsYlJcKN+ZntDt544K9mNEdBCrMvtG6f+luwoZExNCTP9+Trn/PPu+b5+nd6+K5MMfZ5BeUMlfrhnLwilx/PvmSWQVV3Pt0+soqqz/ej3bWfF9owjJsaYOa000tZ5NRKTvckzNZRERB6uobeJ7L28m2M+bx26cQKi/D0G+XqzaU8wFo50zWtMVlmVx07PrKaps+PpYP29PooN9SYoO4hcXJJMYFXjSNfe9s53GFht/vmYcnh6mRzHMT4nm8ZVZHK5q6PW1Si02i9++v4tpiRGnnPJYVFnP1pxyfjJ/uNPiGREdRGxoP5anF3Hd5M5vnwCwfHcRz391kFunDeXc5Nbkb/aIKF64bTLfeX4j1zy5Fn8fT0YNCiakX99Zz3bUzVOHEuznTYo21RYR6bM00iYibsdms7jnja0cKq/jiZtSiQryw9vTg+lJEazIOOz00u6dkXekjqLKBm6fHs+j10/gVxeN5IazhzA6JoQNB8q46NHVPL4yi+YW29fXvLkpl5WZh7nvgmTiIwJ6HMOC8YOwWfDh9kM9vldX7T9czYtrs7njpc38a0VWu31SVtPIT9/aBuDUZNsYw7yR0azeW0Jt48lFRNpTWFHPz/67jZSBwdx3YfJx56YkhPPKd6dQUddERmFVn5saeVR0sB93zBqGRw//SCAiIq6jpE1E3M4/PtvLiszDPHBJChPjvpmONntEJIWV9WQWVbkwulbb8soBuGz8IC4bN4jvzkzg15ek8NgNqSz78UzmJkfxp08yueLxr0gvqCTvSC2//yCdqQnh3Dx1qENiSIoOYuTAYJZs6/2kLb2wtQ8mx4fx56WZ/PSt7TQ0txzXZnP2ES5+dDXrD5Tx8JVjur1+r7Pmp0TT0GxjTSe3Qmix/3GgvsnGP2+YgK+X50ltxg/uz+uLpjA5PowrJsQ6OmQREZFOUdImIm7l84wi/vHZXq5KjeWmKXHHnZs9orW0+4oM16/jSsspx8fLg+QBJ085iwry44mbJvL4jakUVNRx6T/XcNMz67Esiz9dPdahIx4Lxg9ia0452aU1DrtnZ6QXVOLlYXjpO5O5d95w3t6Sx8JnNlBW04hlWTy75gDXPrUWL0/DO3ee0+Upi90xOT6MID8vlqd3rorkEyuzWLe/jN8uGMWwyMB2240cGMybd0zt8tYMIiIijqKkTUTcRm5ZLfe+0TpV7Q9XjMaY45Ob6GA/Rg4MZmVm94pNOFJabjmjBwXj49X+j9GLxgxk2b2zuHTcIA6W1vLrS1IYHObv0DguHTcIgPfSene0Lb2gksSoQHy9PPnRvCQevX4CaXnlXP6vL1n00mZ+/8Fu5iRH8cEPZjA6JqRXYvL29ODc5CiW7S6isdnWYdsjNY3847O9XDx2INdM1AiaiIi4NyVtIuIWGppbuPvVLdhsFk/clIqf98lT1aB1iuTm7CNUurD0f1OLjR35FYwffOpqfKEBPvzt2vFsvn+eU0abYvr3Y3J8GO+m5ffqWr+MgipGHlPY4rJxg3h90RRqG5v5PKOYX16UzNMLJ/Z64Y7Lxg3iSG0Tq/Z0PBr78c5Cmlos7pw17KQ/DoiIiLgbJW0i4hb+8GE62/Iq+PM144gLb79Ix5wRUa2l/zu5bskZMguraGi2MW5w50eQwgOdV91xwfhB7Dtc02sbbR+paaSwsp7kAcevUUsdEsrHP5rJ0ntmsGima5KhmcMjCQ/wYfHWvA7bvZuWz7DIAEZpyqOIiPQBStpExOXe23aIF9dmc/v0+FNWGEwd0p8gPy9WunB/srTc1iIkEzox0tYbLho9EC8Pw5JemiKZXtiaHI5so4R8ZJAviVHOLTjSEW9PDy4dN4jl6cVU1LU9GnuovI4NB8pYMD5Go2wiItInKGkTEZfKKq7mvre3MzEulF+cUHK9LV6eHsxMimTlnmKXlf5Pyy0nLMCHwWHO2Si6q0IDfJg1PJL30g5hszn/a5Je0Fo5Mnmg65KzjlyZGkNjs42PdhS0ef59e7XNy+zrAUVERNydkjYRcZnaxma+/8pm+nl78q8bUvH27NyPpFkjIimqbPg6eeht23LLGT+4v1uN0iyYEENhZT0bDpY5/bkyCiqJCPQhKsjP6c/VHWNiQhgWGcDiLfltnl+Sdojxg/sz1AF75YmIiPQGJW0i4jL/+Gwve4ur+cd1ExgQ0vkEYPbwSABWuKCKZFV9E1mHqxkX27/Xn7sj80ZG4e/j2StTJNMLK9ucGukujDFcmRrLhoNl5JbVHndub1EVuwsqWTBeo2wiItJ3KGkTEZdZtquImUmRTE+K6NJ1UcF+jBoUzCoXrGvbnleBZcH4Ie6VtPn7eHFeSjQf7Sg4Zbn7nmhusbGnqPqkIiTu5vIJMQAs3nr8aNuStEN4GLh47EBXhCUiItItStpExCXyy+vYX1LDjC4mbEfNHhHJ5pwj7RabcJajRUjGxfbO3mNdsWB8DBV1py533xMHSmpobLa59UgbtG6FMCUhjMVbv9kKwbIslmzLZ1pihNtO7RQREWmLkjYRcYk1e1sTixlJkd26fs6IKFpsFmt6ufR/Wm458REB9Pf36dXn7YzpSRGEBfiwJK3ttVyOcHRbgeQB7p20AVw5IZYDJTVfJ9pbc8vJLatjwfgYF0cmIiLSNUraRMQlVu8tISrIl+HRgd26fvzg/gT7ebGyF9e1WZZFmr0IiTvy9vTg4jEDWZ5eRG1js1OeI6OwCm9PQ2JU9/qtN104ZgC+Xh5fT5FcsjUfHy8Pzh8V7eLIREREukZJm4j0OpvN4susEqYnRXS7AqOXpwczh0eyIvMwTS3OW8N1rEMV9RyuanDbpA1aE5X6JhurnTQCmV5QybDIQHy83P/XR5CfN/NTonl/2yHqGlv4YHsB80ZGEeTn7erQREREusT9f+uKyGln16FKjtQ2dXs921FXpcZSUt3AO1vyHBRZx7bZp9m5c9J21tAwgv28WLa7yCn3zyiocvv1bMe6KjWWI7VN/OGj3ZTWNGpqpIiI9ElK2kSk163Oal3PNi2xZ0nb7BGRjBvcn39+nuXUiolHpeWW4+Pp4babSkPrFMk5yVF8nlFMi4M32j5S00hhZT0j3fj1n2hGUgQRgT68vC6HID8vZo/o3hpKERERVzpl0maMGWyMWWGMSTfG7DLG/Mh+PMwYs8wYs9f+OdR+3BhjHjXGZBljthtjUp39IkSkb1mzt4TkAUE9ruBnjOGeeUnkHanj7V4YbUvLKSdlUDC+Xp5Of66emDcymrKaRrbmHOn0NQdKanhy1T6uePxLUn+/jAMlNSe1Se9DRUiO8vL04NJxrXuyXTR6oNv3nYiISFs6M9LWDPzEsqyRwBTgLmNMCnAf8JllWUnAZ/Z/A1wIJNk/FgFPODxqEemz6hpb2HTwSI+nRh41e3gk4wf35zEnj7Y1t9jYkV/h1lMjj5o1IhIvD8Oy9I6nSOYdqeWvn2Zy3t9WMecvK3n44wyaWywam2387v1dJ7VPL6wC6FPTIwGuO2sI/j6eXDt5sKtDERER6ZZTJm2WZRVYlrXF/rgKSAdigAXAC/ZmLwCX2x8vAF60Wq0D+htjtIupSCdZlkVOaS1ZxdXHfew/XP31flN92foDpTS22JjezVL/Jzo62pZfXsdbm3Mdcs+27Cmqpq6phQlutql2W4L9vJmSEM7yDta1tdgsbvj3ev61IotQfx8euCSFNb+Yw/s/mM6P5iaxIvMwn52Q9KUXVBIR6EtkkK+zX4JDjRgQxO7fXUDqkFBXhyIiItItXl1pbIwZCkwA1gPRlmUVQGtiZ4yJsjeLAY5955RnP1Zwwr0W0ToSx5AhQ7oRusjp6d20fO59Y1ub565MjeGRb43v5Ygca83eEnw8PZg8NMxh95w1PJIJQ/rzr8+zuHpirFOmwH2zqbb7J20A81OiefC9Xew/XE1C5Mnl+ZftLiKnrJbHb0zlojHH/13t29OG8samXH77/m6mJUbg59369cworOxT69lEREROF51O2owxgcDbwD2WZVV2UKa7rRMnDQ9YlvU08DTApEmT+v7wgYgDWJbF018cIDEqkB/OTTru3Np9pby2IYfLxg1i9oiodu7g/tZklTBpaCj9fByXWBljuHfecG5+bgNvbspj4ZS4bt2nrrGFXYcqSC+opLHl+B9LS3cVEurvTVy4vyNCdrq5I6N48L1dLE8vYlEbSdtzXx4gNrQf548acNI5b08PfnPpKG56dj3PrN7P3ecm0dxiY09RNd8+Z2gvRC8iIiLH6lTSZozxpjVhe8WyrHfsh4uMMQPto2wDgaM73OYBxy4ciAUOOSpgkdPZuv1lpBdU8serxnCZvXjCUeePimbDgVLuf3cnn947E3+fLg2Uu4XiynoyCqv4+QUjHH7vGUkRTIwL5fEVWXxrUudG26obmvlw+yHScivYlltOZlFVhxUXF4wf1O195XpbbKg/IwcGs3x3MYtmDjvu3M78CjYcKOP+i0fi6dH265meFMGFowfw2IosrkiNpaahmcZmG8kDNNImIiLS2075rs+0vkN5Fki3LOuRY069B9wCPGz/vOSY43cbY14HzgYqjk6jFJGOPfflAcICfNrcS8rXy5P/vWIM1z69jr8v38svLxrpggiPt3hrHsWVDdwxa9ipG9M6ygYw00Hr2Y51dLTtpmfX88bGXG6eOvSU1/zkzTSW7ioi2M+LcYP78/2RwxgX25/RMSFtjgQG+fatRHn+yCgeW5FFWU0jYQE+Xx//z5cH8ffx5JpJHRfm+NXFI1mRWcz/fpjOeaOigb5XhEREROR00Jl3INOAhcAOY0ya/dgvaU3W3jTGfAfIAa6xn/sIuAjIAmqBWx0aschpKqe0luXpRdw9J/HrNUQnOjshnOsnD+bZNQe4bNwgRseE9HKU39iZX8HP3tpOs80ipJ83100+9drUNXtLCAvwIcVJb/ynJYZz1tBQ/rUii29NGtzu1xFaKycu213EopkJ/M+FyX1mBK0r5qVE8+jnWazIKOaqibEAFFfV8/62Q1w/eTAh/bw7vD421J/vz07kkWV7KKqsx9vTMKyNqZYiIiLiXJ2pHrnGsixjWdZYy7LG2z8+siyr1LKsuZZlJdk/l9nbW5Zl3WVZ1jDLssZYlrXJ+S9DpO97/quDeBrDTadYj3XfBSMJ9ffhl4t3OHzz5M5qaG7hJ29uIyzAh6kJ4TywZNfXhTraY1kWq7NKOGdYOB7tTMnrqaOjbUWVDbyxseNKkq+uzwHglnOGnpYJG8DoQSFEB/uy/JgqkK+sy6Gxxca3p8V36h6LZiYwJMyfTdlHGBYZiI9XZ3aKEREREUfSb18RN1BV38Sbm3K5ZOxAooM73nA6xN+bBy9NYXteBS98dbB3AjzB35btJbOoij9eNZbHb0wlMsiXO1/eTEl1Q7vXZBZVcbiqwWH7s7Vn6rBwJseH8fjKLOqbWtps09Dcwhsbc5k7MpqY/v2cGo8reXgY5o2MZtWew9Q3tdDQ3MIr67OZmxxFfERAp+7h5+3Jry9JAXDaCKmIiIh0TEmbiBt4a1Me1Q3N3Da9c6Mfl4wdyJwRkfzl00zyy+ucHN3xNmeX8fQX+7jurMHMSY4iNMCHpxZOpKymkR+8upXmlrY3uF6zt3U9m6P2Z2vPsaNtr23IabPNRzsKKK1p5Oap3asy2ZfMS4mmtrGFtftLeX9bASXVjdzayVG2r+8xMopfXpTMt6cNdU6QIiIi0iElbSIu1mKzeP6rg0yMC2VsJ/cAM8bwuwWjsSx4cMlOJ0f4jdrGZn7y5jYG9e/H/fYfDoFgAAAbR0lEQVTRF4DRMSH84YoxrN1fyp+WZrZ57eq9JSREBvTKyNbUYeGcHR/G4yv3tTna9tLabOIjApg2zLmjfu5gakI4/j6eLNtdxHNrDjA8OpBpieFduocxhkUzh3X6/6eIiIg4lpI2ERf7PKOYnLJabuvi6MfgMH9+MDeR5enF7DpU4aTojvfHjzM4WFrLn68eR+AJlRSvnhjLzVPjePqL/Xyw/RDFVfUs3VXIwx9ncO1Ta1mTVcKMxN5Lku6dP5zDVQ28sv740bad+RVsySnnpilxTltb5078vD2ZmRTJ25vz2F1QyW3T4k/bNXwiIiKnKyVtIi723JoDDArx43x7SfWuuHFyHH7eHry8LtsJkR3vy6wSXlibza3ThjJ1WNsjNfdfnMLEuFB+8NpWJv/hM+54aTPPrN5PfVMLC6fEsaiTWwM4wpSEcKYmhPPEyn3UNX4z2vbS2mz8vD242l5N8UwwLyWahmYbof7eXD7h5O0kRERExL31rU2HRNxMU4uNFpvVYWn5juw+VMna/aXcd2EyXp5d/xtKiL83C8bF8O7WQ9x34chTlnDvru155dz16hYSIgP4xQXJ7bbz8fLgiRtT+cdne4mPCGDCkP6MGhTS7a9PT907fzjfemotr6zP5vYZCVTUNrFkWz6Xj49x2tfKHZ2bHIWftwcLpw51WV+IiIhI92mkTaQHfvrWNi7555p2qxR2ZOPBMu5+bQv9vD257qyONznuyMKpcdQ1tfD25rxu36MjGw6UccO/1xPk58Xz3558yjf9UcF+/OGKMdw+I4GJcWEuTRImx4cxLTGcJ1fto7axmbc251LfZGPhGVCA5FhhAT6s+OlsfjQ3ydWhiIiISDcoaRPppvqmFj7dVURWcTX/WpHV6esq6pr45eIdXPPkWhqabPz75kn09/fpdhyjY0JIHdKfl9dlY3Pwvm2r9hzm5ufWEx3sy1t3nMOQcH+H3r833DtvOCXVjby0NptX1ueQah/9O9MMDOmH5xmwhk9EROR0pKRNpJu+2ldCXVMLSVGBPLFyH5mFVR22tyyLj3cUMP+RVby+IYfbp8fz6b0zme6AfcsWTo1jf0kNX+4r6fG9jvpkZyHffWETCRGBvHHHVAaEdLx/nLuaNDSMGUkR/PXTPRwoqeHmqUNdHZKIiIhIlyhpE+mmZbuLCfT14uXbzybIz4tfLt7R7khXbWMzd768hTtf2UJkkC9L7prO/ZekEODrmGWlF40ZSFiADy+tdUxBksVb87jr1S2Mignmte9OISLQ1yH3dZV75g2nscVGeIAPF44Z4OpwRERERLpEhUjE7TQ223hi5T4q65tOOjdzeCSzhjt3c+bOsNksPksvYtbwSKKD/bj/4hR+8tY2Xt2Qw01Tjl8vVdPQzG3Pb2TjwTLuuzCZ26fHd6voSEd8vTy59qzBPLVqH/nldd3eC23f4Woe+XQPH+4oYEpCGM/cctZJpf37oolxoXx3RjxJUUH4eqkQh4iIiPQtff/dmJx23tt2iL8t30OAj+dx+0k1tdh4ce1BXl80hYlxYa4LENieX0FxVQPzUqIAuDI1hne25vHHjzOYnxJNdHDrVMLqhmZu/c8GtuSU8/frJnDZuEFOi+nGs4fw1Kp9vLo+m5+d336Fx7YcKq/jH8v38t8tefh6efDDcxP5/pzE06rS4K8uTjl1IxERERE3pKRN3IplWTy35gDDowNZes/M45K2itomLvvXGu58eQsf/HA6UUGuW2O1fHcRnh6GOSNakzZjDH+4fAzn//0LfvPeLp64aSKV9U3c8twGduRV8M/rJ3DRmIFOjSk21J9zk6N5Y2MuP5yb1KkRpYq6Jv752V5eXJcNFtw8NY675iT2+emQIiIiIqcTrWkTt7L+QBm7Cyq5dVr8cQkbtO5J9uRNE6mqb+buV7bS1GJzUZSwPL2ISXGhx1V9HBoRwA/nJvHxzkLe3pzHwmfWszO/gsduSHV6wnbUwqlxlFQ38snOwk61//l/t/HclwdYMG4Qn/90Fg9eOkoJm4iIiIibUdImbuU/Xx4g1N+bKybEtHl+5MBgHr5qDBsOlvG/H6X3cnStcstqySisYn5K9EnnFs1MYER0ED95axvpBVU8ceNELhjde4UvZiRGMDTcnxc7UZBk3+FqPt1dxF1zEvnzNeOIDe175fxFREREzgRK2sRt5JTW8unuIq6fPKTDtVQLxsfwnenx/OfLg7y7Nb8XI2y1PL0IoM2kzdvTgz9ePZbEqECeWjiReW20cSYPD8NNU+LYnH2ErTlHOmz7zOr9+Hh6cMs5Q3snOBERERHpFiVt4jZeWHsQT2NYODXulG3vuzCZs+PDuO+d7ew+VOn84I6xbHcRSVGBxIUHtHl+/OD+LP/xLOYkR/VqXEdde9ZgIgJ9+c37u9vdgqC4qp63t+Rz1cRYTYcUERERcXNK2sQtVDc08+bGXC4aM5CBIacuV+/t6cFjN6TSv58Pd7y8iZqG5l6IsrUYyvoDZb0+gtYVQX7e/M+FyWzLLee/W/LabPPCVwdparHx3RkJvRydiIiIiHSVkjZxC//dlEtVQzO3Thva6Wsig3x55FvjyC2r48PtBc4L7hgr9xTTYrOYN9J9kzaAKybEkDqkP3/8OIOKuuP3u6tpaObldTmclxJNfETbo4UiIiIi4j6UtInL2WwWz391kAlD+jNhSGiXrp06LJz4iADebmdEydGWpxcTEejD+MH9e+X5usvDw/C7BaMpq23k78v3HHfujY25VNQ1ccesYS6KTkRERES6QkmbuNyKzGIOltZy27T4Ll9rjOHKCTGsP1BG3pFaJ0T3jcZmGyszipmbHI2nhzn1BS42OiaE6ycP4cW12WQWVgGtG5Q/u+YAZw0NJbWLCbKIiIiIuIaSNnG55748wMAQv26Xxr/cvj2AsytJbjhQRlVDs1uvZzvRz84bQZCfFw++txPLsvhoRwH55XUsmqlRNhEREZG+QkmbuFRGYSVfZpWycGoc3p7d++84OMyfs+PDeGdLPpbVdrXEnjia7Px6yU78vD2Ynhjh8OdwltAAH3563gjW7S/jg+0FPLVqP8MiA5jrosqWIiIiItJ1Xq4OQE5WUdvEjc+uI6u4+qRzIwYE8+Jtkwnp5+2CyBzv5XXZ+Hp5cP1ZQ3p0n6tSY/n529tJyy3v8rq49liWxZqsEv68NJPteRUkRgXy9MJJ9PNpfw85d3T95CG8tiGH+97eTk1jCw9fOQaPPjC9U0RERERaKWlzM5Zl8ct3d5BRUMXNU4fi7fnNm+vGFhsvrc3mx2+k8e+bJ/X5N961jc0s2XqIi8YMJDTAp0f3unDMAH69ZCfvbMl3SNK2NecIf/okk7X7S4np348/Xz2WK1Nj+8RathN5ehh+e9korn5yLZFBvl9PJxURERGRvkFJm5tZvDWfD7cX8LPzR3DXnMSTzsdHBPDAkl388/MsfjQvyQUROs6H2wuoamjmurMG9/heQX7enDdqAO9vP8SvL0nBx6t7Uy33FFXx56WZLNtdRHiADw9cksKNU4bg69W3RtdONGloGL++JIXY0H74efft1yIiIiJyplHS5kZyy2p5YMkuJsWF8r12yrEvnBJHWk45f/9sD2Nigzk3ue8UxTjR6xtzSYgMYHJ8mEPud2VqDO9vO8SKzGLOH9W1oia5ZbX8bfkeFm/NJ9DHix/PH85t0+MJ9D19vkW+M73r1TlFRERExPVUiMRNtNgsfvxmGgB/u3Z8u9PwjDH84YoxjBwQzD2vp3GwpKY3wzzJGxtzmPfIKj7dVdil6/YUVbE5+wjXnTUYYxwz5XBGYgQRgb6804U922oamnlwyU7O/etKPtxewHdnJPDFz+fww7lJp1XCJiIiIiJ9l5I2N/Hkqn1sPHiE3y0YxeAw/w7b9vPx5KmFEzHG8L2XN1Pb2NxLUR6vrKaRhz5MJ7u0hkUvbebOlzdTXFnfqWtf35CLt6fhytRYh8Xj5enB5eMH8XlGMUdqGjt1zVNf7OfFddlcPTGWlT+bzS8vGtnj9XUiIiIiIo6kpM0NbM8r52/L9nDx2IFc0ckiEYPD/Hn0+glkFlVx39s7nFLq/lT+tmwPtY0tvHf3dH52/gg+yyhm7iOreGV9NjZb+/HUN7XwztY85qdEExHo69CYrkyNpanF4oPth07Z1rIslqTlM21YBP935VgGhvRzaCwiIiIiIo6gpM3F6hpbuOf1NCKDfPnfy8d0aargrOGR/PS8Eby37RDvbHHuxtInyiys4pX12SycEsfIgcHcNSeRpffMZExMCL9avJNvPbWWvCO1bV67dFch5bVNXNfDMv9tSRkUTPKAIN7uxNcjLbec7NJaLhs/yOFxiIiIiIg4ipI2F/vjJxnsL6nhr9eMI8S/63uv3TlrGONiQ/jrp5nUN7U4IcKTWZbFQx/uJsjPmx/N/aaCZXxEAK/cfjZ/uWYcmUVV3PTMekqqG066/vUNucSG9nPaJtVXpsaQllvO/sMn73N3rCVph/Dx8uCC0V0rWiIiIiIi0puUtPVQZmEVNz6zjkc+zWTXoYouTVP8al8Jz391kG+fM5RzupnAeHgYfnFhMocq6nlpbXa37tFVKzKLWb23hHvmJZ20/ssYw9UTY3n+1rMorKznluc2UFXf9PX5gyU1rN1fyrWTBjttn7kF42Pw9DC82MHXo7nFxgfbC5ibHEWw3+mxUbmIiIiInJ6UtPWAzWZx3zvb2XTwCI+tyOLiR9cw408r+P0Hu9lwoKzDdV3VDc38/L/bGRruz88vGNGjOM4ZFsGs4ZE8tiKLirqmU1/QA00tNh76IJ2EyABumhLXbruJcWE8cdNEMgurWPTi5q9HAV/fmIuHgWsm9XxvtvZEB/txVWoMr27Ioaidwihf7SulpLqBBZoaKSIiIiJuTklbD7yxKZetOeX835Vj2PCrefzxqjEkRQXy0tpsvvXUWm58Zj0VtW0nUX/4MJ1D5XX89Vvj8PfpeWn5X1yQTGV9E0+u2tfje3XkpbXZ7C+p4f6LR+Lt2fF/nzkjovjLNeNYu7+Ue15Po76phf9uzuPc5CgGhPg5Nc675yTRYrN4YmXbX48laYcI8vVi9ogop8YhIiIiItJTStq6qbS6gYc/zmByfBhXTIghItCXa88awn9unczmX8/j9wtGsSm7jKue/Oqkghyr9hzmtQ05fHdGAhPjHLOxdMqgYC4fH8Nzaw5QWNG5svtddaSmkb8v38OMpAjmdDLZuXxCDA9cksInuwq55sm1lFQ3OKUAyYmGhPtz5YQYXtuQc9I2BPVNLSzdVcgFowfg5+3p9FhERERERHpCSVs3PfxxBjUNzTx0+eiTKj4G+XmzcOpQXrztbIor67ni8a/YkVcBQEVdE7/473aSogK5d/5wh8b04/nDsSz4+/I9Dr0vtE6L/OHrW6ltbOHXl6R0qcrlbdPjuXtOIjvyK4gO9mX2iEiHx9eWu89NpNlm8cQJo4+fZxRT3dDM5Z3cXkFERERExJWUtHXDxoNlvLU5j9tnJDA8OqjddlOHhfP2nefg4+nBt55ay+cZRfz2/V0crm7gr98a5/BRnsFh/tw0JY43N+WSVVzlsPtalsUDS3ayem8Jf7hidIevuT0/OW84/3NhMr9fMBqvU0yrdJS48ACunBDDq+uPH217d2s+kUG+TEkI75U4RERERER6QklbFzW12Lh/8U5i+vfjh3MTT9k+KTqIxXedw7CoAG5/YRPvbMnn+7OHMTa2v1Piu/vcRPx9vPjTJ5kOu+dTX+zntQ25fH/2MK7t5tRGYwx3zBrGeaN6t7z+0dG2J1ftB6CitomVmYe5dOwgPJ1UvVJERERExJGUtHXR818eJLOoigcvTel0AZGoID/eWDSV80cNYHJ8GD84N+nUF3VTWIAP35uVwKe7i9icXdbj+320o4CHP87gkrED+el5Paty6Qpx4QFcMSGGV9ZnU1xVzye7CmhssalqpIiIiIj0GUraumD13sP8bfke5iZHMT8lukvXBvh68cRNE3nzjqn4eDn3y37b9Hiignz5zXu7aW6xddg270gttzy3gb8szWRH3vH7zG3JOcK9b6QxMS6Uv1wzzmn7qjnb3XNaR9ueWrWfd7ceIj4igLGxIa4OS0RERESkU3pea/4MUFrdwEMfprN4az4JkQH8ro3iI+7E38eLBy5N4e5Xt/LC2my+Mz2+zXaWZXHf2zvYcKCMNVklPLYii5j+/ZifEs2UhDB+tXgn0cF+PL1wYp+usjg0IoDLx8fw8rpsGlts/PDcJLfuPxERERGRYylp64BlWbyzJZ+HPtxNdUMzP5ybxPdnD+sTCczFYwbyTnI+f/00k/NHRRMb6n9Smzc35bImq7W4yEWjB7I8vYiluwp5dUMOz391kJB+3vzn1rMID/R1wStwrB+cm8i7aflYFlymqZEiIiIi0ocoaWvHwZIafvXuDr7MKmVSXCj/d+UYkrpRNdFVjDH8bsEo5j/yBQ8s2cWzt0w6bnSpsKKehz5IZ0pCGNefNQQPD8M1kwZzzaTB1DQ0s3pvCfERAQyLDHThq3CcoREBLJwSR3ZpzWnzmkRERETkzKCkrR0Pf5zB9twKHrp8NDdMHtIn13PFhvrzk/OG89CH6Xy4o4BLxraOMFmWxa8W76DJZuOPV4096bUF+HpxwejerfLYG35z2ShXhyAiIiIi0mVK2trxm8tGYQxEB/u5OpQe+fY5Q3k3LZ/fvLebGYmRhPh78962Q3yWUcz9F48kLjzA1SGKiIiIiEgHVD2yHQNC/Pp8wgbg5enBw1eOpaymgYc/yaCkuoHfvLeLCUP6c+u0tguUiIiIiIiI+9BI2xlgdEwIt02L55k1B9h9qIKahhb+fPVYbS4tIiIiItIHnHKkzRjznDGm2Biz85hjYcaYZcaYvfbPofbjxhjzqDEmyxiz3RiT6szgpfPunT+cmP792JZXwY/mJZEY1XeKqoiIiIiInMk6Mz3yeeCCE47dB3xmWVYS8Jn93wAXAkn2j0XAE44JU3oqwNeLf94wge9Mj2fRzARXhyMiIiIiIp10yumRlmV9YYwZesLhBcBs++MXgJXAL+zHX7QsywLWGWP6G2MGWpZV4KiApftSh4SSOiTU1WGIiIiIiEgXdLcQSfTRRMz+Ocp+PAbIPaZdnv3YSYwxi4wxm4wxmw4fPtzNMERERERERE5vjq4e2VZlC6uthpZlPW1Z1iTLsiZFRkY6OAwREREREZHTQ3eTtiJjzEAA++di+/E8YPAx7WKBQ90PT0RERERE5MzW3aTtPeAW++NbgCXHHL/ZXkVyClCh9WwiIiIiIiLdd8pCJMaY12gtOhJhjMkDHgQeBt40xnwHyAGusTf/CLgIyAJqgVudELOIiIiIiMgZozPVI69v59TcNtpawF09DUpERERERERaOboQiYiIiIiIiDiQkjYRERERERE3pqRNRERERETEjSlpExERERERcWNK2kRERERERNyYkjYRERERERE3pqRNRERERETEjZnWrdVcHIQxh4FsV8chAEQAJa4OQlxG/X9mU/+f2dT/Zy71/ZlN/e8+4izLimzrhFskbeI+jDGbLMua5Oo4xDXU/2c29f+ZTf1/5lLfn9nU/32DpkeKiIiIiIi4MSVtIiIiIiIibkxJm5zoaVcHIC6l/j+zqf/PbOr/M5f6/sym/u8DtKZNRERERETEjWmkTURERERExI0paTsDGGOeM8YUG2N2HnNsnDFmrTFmhzHmfWNMsP34UGNMnTEmzf7x5DHXTLS3zzLGPGqMMa54PdJ5juh7Y4y/MeZDY0yGMWaXMeZhV70e6RpHfe8fc+17x95L3JsDf/b7GGOeNsbssf8cuMoVr0e6xoH9f729/XZjzCfGmAhXvB7pvK70vf3cWPu5Xfbzfvbjet/nRpS0nRmeBy444dgzwH2WZY0BFgM/O+bcPsuyxts/vnfM8SeARUCS/ePEe4r7eR7H9P1fLMtKBiYA04wxFzozaHGY53FM/2OMuRKodmaw4nDP45j+/xVQbFnWcCAFWOXEmMVxnqeH/W+M8QL+AcyxLGsssB242+mRS089Tyf73t7HLwPfsyxrFDAbaLJfo/d9bkRJ2xnAsqwvgLITDo8AvrA/XgZ0+JdTY8xAINiyrLVW60LIF4HLHR2rOJYj+t6yrFrLslbYHzcCW4BYB4cqTuCI/gcwxgQCPwYecmiA4lSO6n/gNuD/7Pe0WZalTXj7AAf1v7F/BNhHWYKBQ46MUxyvi31/HrDdsqxt9mtLLctq0fs+96Ok7cy1E7jM/vgaYPAx5+KNMVuNMauMMTPsx2KAvGPa5NmPSd/T1b7/mjGmP3Ap8JnzwxQn6U7//x74K1DbSzGK83Sp/+3f8wC/N8ZsMca8ZYyJ7sV4xbG61P+WZTUBdwI7aE3WUoBnezFecZz2+n44YBljltq/x39uP673fW5GSduZ6zbgLmPMZiAIaLQfLwCGWJY1gda/rL9qn/fc1jxmlR7tm7ra98DXUyheAx61LGt/L8csjtOl/jfGjAcSLcta7JpwxcG6+v3vRevI+peWZaUCa4G/9H7Y4iBd/f73pjVpmwAMonV65P/0ftjiAO31vRcwHbjR/vkKY8xc9L7P7Xi5OgBxDcuyMmgdEscYMxy42H68AWiwP95sjNlH619h8jh+SlwsmiLRJ3Wj7zfZL30a2GtZ1t97PWhxmG70/1nARGPMQVp/Z0QZY1ZaljW796OXnupG/2+mdYT1aNL+FvCdXg5bHKQb/W/sx/bZr3kTuK/3I5eeaq/vaX1/t+rotGdjzEdAKq3r3PS+z41opO0MZYyJsn/2AO4HjlYKjDTGeNofJ9C68HS/ZVkFQJUxZop9XvvNwBKXBC890tW+t//7ISAEuMcVMYvjdON7/wnLsgZZljWU1r/C7lHC1nd1o/8t4H1aixMAzAV293LY4iDd+PmfD6QYYyLtt5gPpPd23NJz7fU9sBQYa1orRXsBs4Ddet/nfjTSdgYwxrxG6y/cCGNMHvAgEGiMucve5B3gP/bHM4HfGWOagRZaqwkdXcx6J60VifoBH9s/xI05ou+NMbG0Vo/LALbYK/4+ZlnWM733SqQ7HPi9L32QA/v/F8BLxpi/A4eBW3vpJUgPOKr/jTG/Bb4wxjQB2cC3e+1FSLd0pe8tyzpijHkE2Ejr9MePLMv60N5O7/vciGn9I5qIiIiIiIi4I02PFBERERERcWNK2kRERERERNyYkjYRERERERE3pqRNRERERETEjSlpExERERERcWNK2kRERERERNyYkjYRERERERE3pqRNRERERETEjf0/UjPiImdiZ2YAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1080x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(ts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "ts_log = np.log(ts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'timeseries' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-0ac78e6bc97e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m#Determing rolling statistics\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mrolmean\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtimeseries\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrolling\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mwindow\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m52\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcenter\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mrolstd\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtimeseries\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrolling\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mwindow\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m52\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcenter\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstd\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;31m#Plot rolling statistics:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'timeseries' is not defined"
     ]
    }
   ],
   "source": [
    "    \n",
    "    #Determing rolling statistics\n",
    "    rolmean = timeseries.rolling(window=52,center=False).mean() \n",
    "    rolstd = timeseries.rolling(window=52,center=False).std()\n",
    "\n",
    "    #Plot rolling statistics:\n",
    "    orig = plt.plot(timeseries, color='blue',label='Original')\n",
    "    mean = plt.plot(rolmean, color='red', label='Rolling Mean')\n",
    "    std = plt.plot(rolstd, color='black', label = 'Rolling Std')\n",
    "    plt.legend(loc='best')\n",
    "    plt.title('Rolling Mean & Standard Deviation')\n",
    "    plt.show(block=False)\n",
    "    \n",
    "    #Perform Dickey-Fuller test:\n",
    "    print ('Results of Dickey-Fuller Test:')\n",
    "    dftest = adfuller(timeseries, autolag='AIC')\n",
    "    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])\n",
    "    for key,value in dftest[4].items():\n",
    "        dfoutput['Critical Value (%s)'%key] = value\n",
    "    print (dfoutput)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_stationarity(data['#Passengers'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'pandas' has no attribute 'rolling_mean'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-9-09d4f1999e56>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mts_log_mv_diff\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrolling_mean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'#Passengers'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;32mlambda\u001b[0m \u001b[0mx\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mmath\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlog\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdiff\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\__init__.py\u001b[0m in \u001b[0;36m__getattr__\u001b[1;34m(name)\u001b[0m\n\u001b[0;32m    260\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0m_SparseArray\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    261\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 262\u001b[1;33m         \u001b[1;32mraise\u001b[0m \u001b[0mAttributeError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34mf\"module 'pandas' has no attribute '{name}'\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    263\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    264\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mAttributeError\u001b[0m: module 'pandas' has no attribute 'rolling_mean'"
     ]
    }
   ],
   "source": [
    "ts_log_mv_diff = pd.rolling_mean(data['#Passengers'].apply(lambda x: math.log(x)), 2).diff(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'test_stationarity' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-10-4513ce263725>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtest_stationarity\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mts_log_mv_diff\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'test_stationarity' is not defined"
     ]
    }
   ],
   "source": [
    "test_stationarity(ts_log_mv_diff)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
