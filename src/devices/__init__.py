# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:06:40 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

The existance of this file in a folder indicates to Python that the folder is a
package. When importing the package, this file is executed once. This allows us
to do:
    
    import folder_name as alias

In this file, we specify each module name with a dot (.) in the front. This
indicates that the module is in the same folder as this file.

By doing:
    
    from .module_name import *

we are allowing the use of all functions/classes/globals of the module using
the folder name:
    
    alias.foo()

Note that any functions/classes/globals which start with an underscore (_) will
not be imported unless specified by __all__.
"""
