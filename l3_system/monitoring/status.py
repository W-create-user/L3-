"""D-owned monitor metadata, independent of A's optional summary fields.

This is not a safety decision and does not replace any action or state gate.
"""

def monitor_status(monitor):
    rules=getattr(monitor,'process_rules',None)
    statistical=getattr(monitor,'statistical_r03_enabled',False) is True
    detector=getattr(monitor,'detector',None)
    return {'schema_version':'DMonitorStatus/1.0','r03_enabled':getattr(rules,'enabled',False) is True,
        'rule_version':getattr(rules,'version',None),'rule_environment':getattr(rules,'environment',None),
        'statistical_r03_enabled':statistical,'statistical_label_scope':getattr(detector,'label_scope',None) if statistical else None,
        'execution_authorized':False,'provenance':'D monitor instance; not an A summary or health assessment'}
