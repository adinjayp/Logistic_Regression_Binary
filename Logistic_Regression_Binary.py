{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ADIAX\\AppData\\Local\\Temp\\ipykernel_18500\\2690998774.py:2: DeprecationWarning: \n",
      "Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),\n",
      "(to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)\n",
      "but was not found to be installed on your system.\n",
      "If this would cause problems for you,\n",
      "please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466\n",
      "        \n",
      "  import pandas as pd\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm import tqdm\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.datasets import load_breast_cancer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class LogisticRegression:\n",
    "    def __init__(self, learningRate, tolerance, regularization=False, lambda_param=0.1, maxIteration=50000):\n",
    "        self.learningRate = learningRate\n",
    "        self.tolerance = tolerance\n",
    "        self.regularization = regularization\n",
    "        self.lambda_param = lambda_param\n",
    "        self.maxIteration = maxIteration\n",
    "        \n",
    "    def datasetReader(self):\n",
    "        breast_cancer = load_breast_cancer()\n",
    "        X, y =  breast_cancer.data, breast_cancer.target\n",
    "        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "        return X_train, y_train, X_test, y_test\n",
    "    \n",
    "    def normalize_data(self, X):\n",
    "        # Normalizing each feature (column-wise)\n",
    "        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)\n",
    "        return X_normalized\n",
    "    \n",
    "    def normalize_train_test_data(self, X_train, X_test):\n",
    "        # Normalize training data\n",
    "        self.X_train = self.normalize_data(X_train)\n",
    "        # Normalize test data\n",
    "        self.X_test = self.normalize_data(X_test)\n",
    "    \n",
    "    def addX0(self, X):\n",
    "        return np.column_stack([np.ones([X.shape[0], 1]), X])\n",
    "    \n",
    "    def sigmoid(self, z):\n",
    "        return 1 / (1 + np.exp(-z))\n",
    "    \n",
    "    def decision_boundary(self, X, w):\n",
    "        # Plot decision boundary\n",
    "        if X.shape[1] == 2:\n",
    "            a = -w[0] / w[1]\n",
    "            b = -w[2] / w[1]\n",
    "            x1 = np.linspace(-5, 5, 100)\n",
    "            x2 = -(w[0] + w[1] * x1) / w[2]\n",
    "            plt.plot(x1, x2, label='Decision Boundary')\n",
    "            plt.scatter(X[:, 0], X[:, 1], c=self.y_train, cmap='viridis')\n",
    "            plt.xlabel('Feature 1')\n",
    "            plt.ylabel('Feature 2')\n",
    "            plt.legend()\n",
    "            plt.show()\n",
    "        else:\n",
    "            print(\"Cannot plot decision boundary for datasets with more than two features.\")\n",
    "        \n",
    "    def costFunction(self, X, y):\n",
    "        z = X.dot(self.w)\n",
    "        h = self.sigmoid(z)\n",
    "        epsilon = 1e-5  # to prevent log(0) case\n",
    "        regularization_term = 0\n",
    "        if self.regularization:\n",
    "            regularization_term = (self.lambda_param / (2 * X.shape[0])) * np.sum(self.w[1:]**2)  # Exclude bias term\n",
    "        cost = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)) + regularization_term\n",
    "        return cost\n",
    "    \n",
    "    def gradient(self, X, y):\n",
    "        z = X.dot(self.w)\n",
    "        h = self.sigmoid(z)\n",
    "        grad = np.dot(X.T, (h - y)) / y.size\n",
    "        if self.regularization:\n",
    "            grad[1:] += (self.lambda_param / X.shape[0]) * self.w[1:]  # Exclude bias term\n",
    "        return grad\n",
    "    \n",
    "    def gradientDescent(self, X, y):\n",
    "        cost_sequences = []\n",
    "        last_cost = float('inf')\n",
    "        for i in tqdm(range(self.maxIteration)):\n",
    "            self.w = self.w - self.learningRate * self.gradient(X, y)\n",
    "            cur_cost = self.costFunction(X, y)\n",
    "            diff = last_cost - cur_cost\n",
    "            last_cost = cur_cost\n",
    "            cost_sequences.append(cur_cost)\n",
    "            if diff < self.tolerance:\n",
    "                print('The model stopped: Converged')\n",
    "                break\n",
    "                \n",
    "        self.plotCost(cost_sequences)\n",
    "        return \n",
    "    \n",
    "    def plotCost(self, error_sequences):\n",
    "        plt.plot(error_sequences)\n",
    "        plt.xlabel('Iteration')\n",
    "        plt.ylabel('Error')\n",
    "        plt.title('Cost Function')\n",
    "        plt.show()\n",
    "                        \n",
    "    def predict(self, X):\n",
    "        z = X.dot(self.w)\n",
    "        return np.round(self.sigmoid(z))\n",
    "    \n",
    "    def evaluate(self, y, y_hat):\n",
    "        accuracy = np.mean(y == y_hat)\n",
    "        precision = np.sum((y == 1) & (y_hat == 1)) / np.sum(y_hat)\n",
    "        recall = np.sum((y == 1) & (y_hat == 1)) / np.sum(y)\n",
    "        return accuracy, precision, recall\n",
    "    \n",
    "    def fit(self):\n",
    "        X_train, y_train, X_test, y_test = self.datasetReader()\n",
    "        self.normalize_train_test_data(X_train, X_test)  # Normalize train and test data\n",
    "        self.X_train = self.addX0(self.X_train)  # Add intercept term\n",
    "        self.X_test = self.addX0(self.X_test)    # Add intercept term\n",
    "        if self.regularization:\n",
    "            print('Solving using Gradient Descent Regularization: Enabled')\n",
    "        else:\n",
    "            print('Solving using Gradient Descent Regularization: Disabled')\n",
    "        self.w = np.random.randn(self.X_train.shape[1])  # Initialize weights randomly\n",
    "        self.gradientDescent(self.X_train, y_train)\n",
    "        # Prediction and Evaluation on Test Data\n",
    "        y_hat_test = self.predict(self.X_test)\n",
    "        print(\"\\nPredicted classes on test data:\", y_hat_test)\n",
    "        print(\"Actual classes on test data:\", y_test)\n",
    "        accuracy_test, precision_test, recall_test = self.evaluate(y_test, y_hat_test)\n",
    "        print('\\nEvaluation for test data:')\n",
    "        print('Accuracy:', accuracy_test)\n",
    "        print('Precision:', precision_test)\n",
    "        print('Recall:', recall_test)\n",
    "        self.decision_boundary(X_train, self.w)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Solving using Gradient Descent Regularization: Enabled\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/50000 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  4%|▍         | 2164/50000 [00:01<00:25, 1860.14it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The model stopped: Converged\n"
     ]
    },
    {
     "data": {
      "image/png": "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EFIINwAAIKQQbgAAQEgh3AAAgJBCuAmQefPmKTMzUxEREcrOztbatWvNLqlbePjhh2WxWPyW8847z7e9vr5e06ZN01lnnaWYmBh9//vfV2lpqd9r7NmzR9dee62ioqKUmpqqn//852psbAz2RwkJ77//vq6//nr16NFDFotF//rXv/y2G4ahWbNmKSMjQ5GRkcrJydGXX37p1+bQoUO69dZbFRcXp4SEBP3kJz9RTU2NX5v//ve/GjNmjCIiIuR0OvXkk0929EcLCSc6Prfffnur/56uuuoqvzYcn45TUFCgkSNHKjY2VqmpqRo/fry2bt3q1yZQf9NWrlypCy+8UA6HQ/3799eiRYs6+uMFFeEmAJYsWaL8/HzNnj1bGzZs0LBhw5Sbm6uysjKzS+sWzj//fO3fv9+3rF692rft3nvv1X/+8x+99tprWrVqlfbt26fvfe97vu0ej0fXXnut3G63PvzwQy1evFiLFi3SrFmzzPgoXV5tba2GDRumefPmtbn9ySef1B//+EfNnz9fa9asUXR0tHJzc1VfX+9rc+utt2rz5s1avny53nzzTb3//vu64447fNurqqo0btw49e7dW+vXr9dTTz2lhx9+WC+++GKHf76u7kTHR5Kuuuoqv/+e/v73v/tt5/h0nFWrVmnatGn66KOPtHz5cjU0NGjcuHGqra31tQnE37SdO3fq2muv1eWXX66NGzdqxowZmjx5st5+++2gft4OZeCMjRo1ypg2bZrvucfjMXr06GEUFBSYWFX3MHv2bGPYsGFtbquoqDDCw8ON1157zbeuuLjYkGQUFRUZhmEYS5cuNaxWq1FSUuJr88ILLxhxcXGGy+Xq0NpDnSTjjTfe8D33er1Genq68dRTT/nWVVRUGA6Hw/j73/9uGIZhfP7554YkY926db42//u//2tYLBbj66+/NgzDMJ5//nkjMTHR7/j84he/MAYMGNDBnyi0fPP4GIZhTJo0ybjhhhuOuw/HJ7jKysoMScaqVasMwwjc37QHHnjAOP/88/3ea8KECUZubm5Hf6SgoefmDLndbq1fv145OTm+dVarVTk5OSoqKjKxsu7jyy+/VI8ePdS3b1/deuut2rNnjyRp/fr1amho8Ds25513ns4++2zfsSkqKtKQIUOUlpbma5Obm6uqqipt3rw5uB8kxO3cuVMlJSV+xyM+Pl7Z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Fzr33HMlSeecc46efPJJv9c8dv5OZmamfve73+nOO+/U888/L7vdrvj4eFksFr/3+6bCwkJ99tln2rlzp5xOpyTpz3/+s84//3ytW7dOI0eOlNQUghYtWqTY2FhJ0m233abCwkI9+uijZ/aLARBw9NwAMNWnn36qFStWKCYmxrecd955kpp6S1pkZWW12vfdd9/VFVdcoZ49eyo2Nla33XabDh48qLq6upN+/+LiYjmdTl+wkaRBgwYpISFBxcXFvnWZmZm+YCNJGRkZKisrO6XPCiA46LkBYKqamhpdf/31euKJJ1pty8jI8D2Ojo7227Zr1y5dd911mjp1qh599FElJSVp9erV+slPfiK3262oqKiA1hkeHu733GKxyOv1BvQ9AAQG4QZA0Njtdnk8Hr91F154of7nf/5HmZmZCgs7+T9J69evl9fr1e9//3tZrU2d0P/4xz9O+H7fNHDgQO3du1d79+719d58/vnnqqio0KBBg066HgCdB8NSAIImMzNTa9as0a5du1ReXi6v16tp06bp0KFDuvnmm7Vu3Tpt375db7/9tvLy8toNJv3791dDQ4OeffZZ7dixQ6+88opvovGx71dTU6PCwkKVl5e3OVyVk5OjIUOG6NZbb9WGDRu0du1aTZw4UZdddplGjBgR8N8BgI5HuAEQNPfff79sNpsGDRqklJQU7dmzRz169NAHH3wgj8ejcePGaciQIZoxY4YSEhJ8PTJtGTZsmObMmaMnnnhCgwcP1l//+lcVFBT4tbnooot05513asKECUpJSWk1IVlqGl7697//rcTERF166aXKyclR3759tWTJkoB/fgDBYTEMwzC7CAAAgECh5wYAAIQUwg0AAAgphBsAABBSCDcAACCkEG4AAEBIIdwAAICQQrgBAAAhhXADAABCCuEGAACEFMINAAAIKYQbAAAQUgg3AAAgpPx/7kBreAslk58AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Predicted classes on test data: [1. 0. 0. 1. 1. 0. 0. 0. 1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 1. 0. 1. 1. 0. 1.\n",
      " 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1.\n",
      " 1. 1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 0. 1. 1. 1. 0. 0. 1. 1. 0. 0.\n",
      " 1. 0. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1.\n",
      " 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 1. 1. 0.]\n",
      "Actual classes on test data: [1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0\n",
      " 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0\n",
      " 1 1 1 0 1 1 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0\n",
      " 1 1 0]\n",
      "\n",
      "Evaluation for test data:\n",
      "Accuracy: 0.9824561403508771\n",
      "Precision: 0.9859154929577465\n",
      "Recall: 0.9859154929577465\n",
      "Cannot plot decision boundary for datasets with more than two features.\n"
     ]
    }
   ],
   "source": [
    "model = LogisticRegression(learningRate=0.1, tolerance=1e-5, regularization=True, lambda_param=0.01)\n",
    "model.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Solving using Gradient Descent Regularization: Disabled\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|          | 0/50000 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  2%|▏         | 1090/50000 [00:00<00:22, 2176.85it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The model stopped: Converged\n"
     ]
    },
    {
     "data": {
      "image/png": "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N6uoqOiY9Vu3bg3d9nz55ZdLkmbNmhW2zSOPPCJJGj16tKSaSyZH9ywMHDhQkkK3jCcmJkpSg2GkqXr06KEPPvhAlZWVoWULFizQzp07m3S8s88+W927d9esWbOOqbP++SUlJUn6+nPxeDy67LLL9MYbb2j79u2h5UVFRXrxxRd1wQUXhC4VAnAGY26AVqhHjx568cUXNXbsWPXp0ydshuL3339fr7zySmjumAEDBmjChAl6+umnVVxcrOHDh2vlypV67rnnNGbMGF188cWSpOeee05PPPGErr76avXo0UNlZWX64x//qJSUlFBAatOmjfr27auXXnpJ3/jGN9S+fXv169fvlMa33Hzzzfr73/+ub33rW/r+97+vrVu36q9//WuoB6ax3G63nnzySV155ZUaOHCgJk6cqKysLG3evFmbNm3SW2+9JUkaPHiwpJrZk0eOHCmPx3PcR0L8+te/VkFBgS644ALddttt8nq9euqpp1RRUaGHHnqoaScO4Pjs3qwFwKZPPvnE3HLLLSYnJ8fEx8eb5ORkc/7555vHHnvMlJeXh7arqqoyDz74oOnevbuJi4sz2dnZZvr06WHbrF271lx33XWma9euxufzmU6dOpkrrrjCrF69Ouw733//fTN48GATHx//tbeFB2/ffuWVV054Hg8//LDp0qWL8fl85vzzzzerV68+7q3gRx9r27ZtRpJ59tlnw5YvW7bMXHrppSY5OdkkJSWZs846yzz22GOh9dXV1eb222836enpxuVyhd0W3tB5rV271owcOdK0bdvWJCYmmosvvti8//77YdsEbwU/+nb6YO3vvPPOCdsBQA2XMYxQAwAAsYMxNwAAIKYQbgAAQEwh3AAAgJhCuAEAADGFcAMAAGIK4QYAAMSUVjeJXyAQ0O7du5WcnByRqeABAIDzjDEqKytT586d5XafuG+m1YWb3bt38xwXAABaqJ07d+q000474TatLtwkJydLqmkcnucCAEDLUFpaquzs7NDv8RNpdeEmeCkqJSWFcAMAQAtzMkNKGFAMAABiCuEGAADEFMINAACIKYQbAAAQUwg3AAAgphBuAABATCHcAACAmEK4AQAAMYVwAwAAYgrhBgAAxBTCDQAAiCmEGwAAEFNa3YMzI6Wi2q99ZRXyuF3KSm1juxwAAFotem4csnFXqS747Tu69ukPbJcCAECr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p+tp8YJjbpjnBgAAO7xN2enxxx/XZ599ps6dO6tbt25KSkoKW7927VpHimuJPDx+AQAAq5oUbsaMGeNwGbHDXdsXRrgBAMCOJoWb+++/3+k6YoaHW8EBALCqSeEmaM2aNfr4448lSWeeeaYGDRrkSFEtmae264aeGwAA7GhSuNm7d6+uvfZaLVmyRGlpaZKk4uJiXXzxxZo3b57S09OdrLFFqX20FPPcAABgSZPulrr99ttVVlamTZs26cCBAzpw4IA2btyo0tJS/fSnP3W6xhaFeW4AALCrST03b775pt5++2316dMntKxv376aPXu2LrvsMseKa4mCMxRzKzgAAHY0qecmEAgoLi7umOVxcXEKBFr347C9Hm4FBwDApiaFm29+85uaMmWKdu/eHVq2a9cu3XHHHbrkkkscK64lcjPPDQAAVjUp3Dz++OMqLS1VTk6OevTooR49eqh79+4qLS3VY4895nSNLYqHxy8AAGBVk8bcZGdna+3atXr77be1efNmSVKfPn2Ul5fnaHEtUXCGYua5AQDAjkaHm6qqKrVp00br16/XpZdeqksvvTQSdbVYwZ4bbgUHAMAOngruMG4FBwDALp4K7jB3aMyN5UIAAGileCq4w7xu5rkBAMAmngrusLpbwem6AQDAhkaHm+rqarlcLt1444067bTTIlFTixZXO4lftZ+eGwAAbGj0mBuv16vf/e53qq6ujkQ9LV6cp6ZJqwg3AABY0eQZipcuXep0LTHBGwo3XJYCAMCGJo25GTVqlO655x5t2LBBgwcPPmZA8be//W1HimuJ4oOXpRhzAwCAFU0KN7fddpsk6ZFHHjlmncvlatVz4AR7biqruSwFAIANTQo3rf3J3yfiddNzAwCATY0ac3P55ZerpKQk9HnmzJkqLi4Off7yyy/Vt29fx4prieK9jLkBAMCmRoWbt956SxUVFaHPv/nNb8JmKa6urtaWLVucq64F8rq5WwoAAJsaFW7MUbPuHv25sd59911deeWV6ty5s1wul+bPn/+1+yxZskRnn322fD6fevbsqblz555SDU4LznNDzw0AAHY06VZwpxw6dEgDBgzQ7NmzT2r7bdu2afTo0br44ou1fv16TZ06VTfffLPeeuutCFd68oLz3DCJHwAAdjRqQLHL5ZKr9vEC9Zc11ahRozRq1KiT3n7OnDnq3r27Hn74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so0GoEkINwCi4tFHH1Vubq5uueUW7dmzR3v27FF2draKi4v1zW9+U4MGDdLq1av15ptvqqioSN///vfD9n/uuecUHx+v9957T3PmzJEkud1u/eEPf9CmTZv03HPPafHixbr77rslScOGDdOsWbOUkpIS+r677rrrmLoCgYCuuuoqHThwQEuXLlVBQYE+//xzjR07Nmy7rVu3av78+VqwYIEWLFigpUuXaubMmRFqLQCngstSAKIiNTVV8fHxSkxMVGZmZmj5448/rkGDBuk3v/lNaNkzzzyj7OxsffLJJ/rGN74hSTrjjDP00EMPhR2z/vidnJwc/frXv9att96qJ554QvHx8UpNTZXL5Qr7vqMtWrRIGzZs0LZt25SdnS1J+stf/qIzzzxTq1at0jnnnCOpJgTNnTtXycnJkqTrr79eixYt0owZM06tYQA4jp4bAFZ9+OGHeuedd9S2bdvQq3fv3pJqekuCBg8efMy+b7/9ti655BJ16dJFycnJuv766/Xll1/q8OHDJ/39H3/8sbKzs0PBRpL69u2rtLQ0ffzxx6FlOTk5oWAjSVlZWdq7d2+jzhVAdNBzA8CqgwcP6sorr9Rvf/vbY9ZlZWWF3iclJYWt2759u6644gr95Cc/0YwZM9S+fXstW7ZMN910kyorK5WYmOhonXFxcWGfXS6XAoGAo98BwBmEGwBREx8fL7/fH7bs7LPP1quvvqqcnBx5vSf/n6Q1a9YoEAjo4Ycflttd0wn98ssvf+33Ha1Pnz7auXOndu7cGeq9+eijj1RcXKy+ffuedD0Amg8uSwGImpycHK1YsULbt2/X/v37FQgENGnSJB04cEDXXXedVq1apa1bt+qtt97SxIkTTxhMevbsqaqqKj322GP6/PPP9fzzz4cGGtf/voMHD2rRokXav39/g5er8vLy1L9/f40bN05r167VypUrNX78eA0fPlxDhgxxvA0ARB7hBkDU3HXXXfJ4POrbt6/S09O1Y8cOde7cWe+99578fr8uu+wy9e/fX1OnTlVaWlqoR6YhAwYM0COPPKLf/va36tevn1544QXl5+eHbTNs2DDdeuutGjt2rNLT048ZkCzVXF5644031K5dO1100UXKy8vT6aefrpdeesnx8wcQHS5jjLFdBAAAgFPouQEAADGFcAMAAGIK4QYAAMQUwg0AAIgphBsAABBTCDcAACCmEG4AAEBMIdwAAICYQrgBAAAxhXADAABiCuEGAADEFMINAACIKf8fbmHCMAqiPL8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Predicted classes on test data: [1. 0. 0. 1. 1. 0. 0. 0. 1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 1. 0. 1. 1. 0. 1.\n",
      " 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1.\n",
      " 1. 1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 0. 1. 1. 1. 0. 0. 1. 1. 0. 0.\n",
      " 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.\n",
      " 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 1. 1. 0.]\n",
      "Actual classes on test data: [1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0\n",
      " 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0\n",
      " 1 1 1 0 1 1 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0\n",
      " 1 1 0]\n",
      "\n",
      "Evaluation for test data:\n",
      "Accuracy: 0.9824561403508771\n",
      "Precision: 0.9726027397260274\n",
      "Recall: 1.0\n",
      "Cannot plot decision boundary for datasets with more than two features.\n"
     ]
    }
   ],
   "source": [
    "model2 = LogisticRegression(learningRate=0.1, tolerance=1e-5, regularization=False, lambda_param=0.1)\n",
    "model2.fit()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".logenv",
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
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
