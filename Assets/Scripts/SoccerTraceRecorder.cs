using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// Opt-in, file-backed trace for offline reward-hacking investigations.
/// Attach to the same GameObject as SoccerEnvController. Not required for training.
/// </summary>
public sealed class SoccerTraceRecorder : MonoBehaviour
{
    [SerializeField] private bool captureEnabled = false;
    [SerializeField, Min(1)] private int sampleEveryFixedSteps = 5;

    private StreamWriter writer;
    private int fixedSteps;

    public bool IsRecording => writer != null;

    private void Start()
    {
        if (!captureEnabled) return;

        try
        {
            string folder = Path.Combine(Application.persistentDataPath, "Loophole");
            Directory.CreateDirectory(folder);
            string file = Path.Combine(folder, "soccer_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture) + ".csv");
            writer = new StreamWriter(new FileStream(file, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 65536));
            writer.WriteLine("event,episode,time_s,team,agent,term,value,agent_x,agent_y,agent_z,ball_x,ball_y,ball_z");
            Debug.Log("[SoccerTraceRecorder] Writing optional reward trace to " + file);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[SoccerTraceRecorder] Could not start optional telemetry: " + ex.Message);
            writer?.Dispose();
            writer = null;
        }
    }

    public void RecordPositions(int episode, float timeSeconds, Vector3 ballPosition,
        IList<SoccerAgent> blue, IList<SoccerAgent> red)
    {
        if (!IsRecording || (++fixedSteps % Mathf.Max(1, sampleEveryFixedSteps)) != 0) return;
        RecordTeamPositions(episode, timeSeconds, ballPosition, blue, "blue");
        RecordTeamPositions(episode, timeSeconds, ballPosition, red, "red");
    }

    private void RecordTeamPositions(int episode, float timeSeconds, Vector3 ballPosition,
        IList<SoccerAgent> agents, string team)
    {
        if (agents == null) return;
        foreach (SoccerAgent agent in agents)
        {
            if (agent == null) continue;
            WriteRow("position", episode, timeSeconds, team, agent.name, "", 0f,
                agent.transform.position, ballPosition);
        }
    }

    public void RecordAgentReward(int episode, float timeSeconds, SoccerAgent agent,
        string term, float value, Vector3 ballPosition)
    {
        if (!IsRecording || agent == null || Mathf.Approximately(value, 0f)) return;
        WriteRow("reward", episode, timeSeconds, agent.team.ToString().ToLowerInvariant(),
            agent.name, term, value, agent.transform.position, ballPosition);
    }

    public void RecordTeamReward(int episode, float timeSeconds, Team team, string term,
        float value, Vector3 ballPosition)
    {
        if (!IsRecording || Mathf.Approximately(value, 0f)) return;
        WriteRow("reward", episode, timeSeconds, team.ToString().ToLowerInvariant(),
            "", term, value, Vector3.zero, ballPosition);
    }

    public void RecordEpisodeEnd(int episode, float timeSeconds, int goals,
        int touches, int passes, int interceptions, Vector3 ballPosition)
    {
        if (!IsRecording) return;
        WriteRow("outcome", episode, timeSeconds, "", "", "goals", goals, Vector3.zero, ballPosition);
        WriteRow("outcome", episode, timeSeconds, "", "", "touches", touches, Vector3.zero, ballPosition);
        WriteRow("outcome", episode, timeSeconds, "", "", "passes", passes, Vector3.zero, ballPosition);
        WriteRow("outcome", episode, timeSeconds, "", "", "interceptions", interceptions, Vector3.zero, ballPosition);
        writer.Flush();
        fixedSteps = 0;
    }

    private void WriteRow(string eventName, int episode, float timeSeconds, string team,
        string agent, string term, float value, Vector3 agentPosition, Vector3 ballPosition)
    {
        if (writer == null) return;
        writer.WriteLine(string.Join(",", new[]
        {
            Escape(eventName), episode.ToString(CultureInfo.InvariantCulture),
            F(timeSeconds), Escape(team), Escape(agent), Escape(term), F(value),
            F(agentPosition.x), F(agentPosition.y), F(agentPosition.z),
            F(ballPosition.x), F(ballPosition.y), F(ballPosition.z)
        }));
    }

    private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static string Escape(string value)
    {
        if (string.IsNullOrEmpty(value)) return "";
        return value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0
            ? value
            : "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private void OnDisable()
    {
        writer?.Flush();
        writer?.Dispose();
        writer = null;
    }
}
