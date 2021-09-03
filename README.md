
# qo

Quick access to code (personalized/collaborative) favorites.

We all have our go-to utils and awesome tools. 
Some of us are annoyed at having to type long dotpaths to the object we want to import.
Some of us probably forget where that particular object is in the first place.

This is to mitigate that. 

Instead do this:

```python
from qo.my_medley import that_object
```

Also have several `my_medley` sets of favorites, since what you'll need depends on the context.

**Word of advice: This tool is meant for the quick-and-dirty development context**

Don't use this for production or any long-term code. It's meant for pulling things together quickly. Once the code matures, you should import "normally".


# Usage

See what modules are available to import

```python
>>> from qo import source_module_names
>>> source_module_names
['tw', 'ca']
```

See what that module is about (if the author cared to say)

```python
>>> from qo import tw
>>> print(tw.__doc__)

A medley of my most frequently used tools.
```

Import it (as a module object)
```python
from qo import tw
# or from qo import tw as my_own_name
```

Inject the module's contents in your current namespace (if you're that type)

```python
from qo.tw import *
```

Inject everything there is in your namespace (but expect unpredictable name collisions)
```python
from qo import *
```



# Inspiration

When I have to do this:

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
```

I go 🤮.

I can never remember what is where. 
Some packages (like `numpy`) percolate the best-of up to the root so that 
these objects can be imported from there. Others (like `sklearn`) don't. 
How do we solve this for ourselves -- and while we're at it, 
handle multiple packages from a same place (`qo`)?

When in fast mode, these kinds of things can really slow one down.
Irrelevant details are detrimental to thinking. 
Where an object is irrelevant when you're modeling. 

I thought this might be more appropriate:

```python
from qo import (
    Ridge, confusion_matrix, LinearRegression, AdaBoostClassifier, RandomForestClassifier,
    SVR, RandomForestRegressor, LogisticRegression, DecisionTreeRegressor, AdaBoostRegressor,
    MLPRegressor, GradientBoostingRegressor, KNeighborsRegressor, StackingRegressor,
    RidgeCV, LassoCV, StandardScaler
)
```

So I did it.


