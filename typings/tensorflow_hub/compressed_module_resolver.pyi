"""
This type stub file was generated by pyright.
"""

from tensorflow_hub import resolver

"""Functions to resolve TF-Hub Module stored in compressed TGZ format."""
LOCK_FILE_TIMEOUT_SEC = ...
_HUB_TF_GOOGLE_CN = ...
_GCS_GOOGLE_CN_TEMPLATE = ...
_COMPRESSED_FORMAT_QUERY = ...
class HttpCompressedFileResolver(resolver.HttpResolverBase):
  """Resolves HTTP handles by downloading and decompressing them to local fs."""
  def is_supported(self, handle): # -> bool:
    ...
  
  def __call__(self, handle):
    ...
  


class GcsCompressedFileResolver(resolver.Resolver):
  """Resolves GCS handles by downloading and decompressing them to local fs."""
  def is_supported(self, handle):
    ...
  
  def __call__(self, handle):
    ...
  

