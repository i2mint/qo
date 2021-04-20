
# qo

Quick access to code favorites.

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




