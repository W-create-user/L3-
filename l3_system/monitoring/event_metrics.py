"""Event evaluation uses raw label intervals, not dilated window truth segments."""
import numpy as np


def evaluate_events(labels,predictions,window_size):
    if type(window_size) is not int or window_size<2:raise ValueError('window_size >=2 required')
    if any(not isinstance(v,(bool,np.bool_)) for v in labels):raise ValueError('explicit boolean sample labels required')
    if any(not isinstance(v,(bool,np.bool_)) for v in predictions):raise ValueError('explicit boolean predictions required')
    if len(labels)<window_size or len(predictions)!=len(labels)-window_size+1:
        raise ValueError('one prediction per complete window required')
    labels=[bool(v) for v in labels];predictions=[bool(v) for v in predictions]
    events=[];index=0;first_available=window_size-1
    while index<len(labels):
        if not labels[index]:index+=1;continue
        start=index
        while index+1<len(labels) and labels[index+1]:index+=1
        end=index;eligible_start=max(start,first_available)
        scorable=eligible_start<=end
        hits=[step for step in range(eligible_start,end+1) if predictions[step-first_available]] if scorable else []
        first_hit=hits[0] if hits else None
        events.append(dict(event_index=len(events),start_record=start,end_record=end,
            scorable=scorable,onset_before_first_window=start<first_available,
            eligible_start_record=eligible_start if scorable else None,
            first_detection_record=first_hit,detected=bool(hits),
            missed=bool(scorable and not hits),delay_records=first_hit-start if hits else None))
        index+=1
    window_truth=[any(labels[i-window_size+1:i+1]) for i in range(first_available,len(labels))]
    total=len(events);scorable=sum(e['scorable'] for e in events);detected=sum(e['detected'] for e in events)
    return {'schema_version':'EventEvaluation/1.0','event_definition':'maximal contiguous true intervals in raw sample labels',
            'events':events,'event_count':total,'scorable_event_count':scorable,
            'unscorable_event_count':total-scorable,'detected_event_count':detected,
            'missed_event_count':sum(e['missed'] for e in events),
            'scorable_event_recall':detected/scorable if scorable else None,
            'window_count':len(predictions),'normal_window_count':sum(not value for value in window_truth),
            'false_alarm_window_count':sum(p and not t for p,t in zip(predictions,window_truth)),
            'all_normal_episode':not any(labels),'all_normal_episode_with_alarm':not any(labels) and any(predictions),
            'delay_unit':'interaction_records; not physical seconds',
            'credit_policy':'only window-end predictions during the raw event interval count; no credit for post-event tails'}


def aggregate_events(reports):
    keys=('event_count','scorable_event_count','unscorable_event_count','detected_event_count',
          'missed_event_count','window_count','normal_window_count','false_alarm_window_count')
    result={key:sum(report[key] for report in reports) for key in keys}
    result['episode_count']=len(reports)
    result['all_normal_episode_count']=sum(report['all_normal_episode'] for report in reports)
    result['all_normal_episode_with_alarm_count']=sum(report['all_normal_episode_with_alarm'] for report in reports)
    denominator=result['scorable_event_count']
    result['scorable_event_recall']=result['detected_event_count']/denominator if denominator else None
    delays=[event['delay_records'] for report in reports for event in report['events'] if event['detected']]
    result['detected_events_with_delay']=len(delays)
    result['mean_delay_records_among_detected']=sum(delays)/len(delays) if delays else None
    result['limits']='Detected-only delay excludes misses; report missed and unscorable counts alongside it. Windows are correlated.'
    return result
