## Purpose

The sample app demonstrates how to simplify [deepstream_nvds_analytics_test_app](../../pipeline_api/deepstream_nvdsanalytics_test_app)
using Flow API.
Flow APIs effectively abstract away the underlying pipeline details, allowing 
developers to focus solely on the goals of their specific tasks in a pythonic style.

## Usage
```
$ python3 deepstream_nvdsanalytics.py <uri1> [uri2] ... [uriN]
```

For URI(s) with special characters like @,& etc, you need to pass the uri within quotes

```
$ python3 deepstream_nvdsanalytics.py 'rtsp://user@ip/cam/realmonitor?channel=1&subtype=0' 'rtsp://user@ip/cam/realmonitor?channel=1&subtype=0'
```