-- Illustrative funnel SQL
SELECT event_name AS step, COUNT(DISTINCT user_id) AS users
FROM events
WHERE event_name IN ('signup','first_prompt','first_output','project_created','project_shipped')
GROUP BY 1;
