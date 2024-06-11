"""
This type stub file was generated by pyright.
"""

"""Internal. Registry holds python objects that can be injected."""
class MultiImplRegister:
  """Utility class to inject multiple implementations of methods.

  An implementation must implement __call__ and is_supported with the same
  set of arguments. The registered implementations "is_supported" methods are
  called in reverse order under which they are registered. The first to return
  true is then invoked via __call__ and the result returned.
  """
  def __init__(self, name) -> None:
    ...
  
  def clear_implementations(self): # -> None:
    """Remove all implementations."""
    ...
  
  def add_implementation(self, impl): # -> None:
    """Register an implementation."""
    ...
  
  def __call__(self, *args, **kwargs):
    ...
  


resolver = ...
loader = ...