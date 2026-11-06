using System;
using System.Diagnostics;
using System.IO;

public static class WinUtilsShim
{
    private static string Quote(string value) { return "\"" + value.Replace("\"", "\\\"") + "\""; }

    public static int Main(string[] args)
    {
        string log = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "winutils-shim.log");
        File.AppendAllText(log, DateTime.Now.ToString("O") + " ARGS " + string.Join("|", args) + Environment.NewLine);
        if (args.Length > 0)
        {
            string command = args[0].ToLowerInvariant();
            if (command == "chmod" || command == "chown")
            {
                File.AppendAllText(log, "  -> skip chmod" + Environment.NewLine);
                return 0;
            }
        }
        string real = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "winutils-real.exe");
        var start = new ProcessStartInfo(real);
        start.UseShellExecute = false;
        start.RedirectStandardError = true;
        start.RedirectStandardOutput = true;
        var quoted = new string[args.Length];
        for (int i = 0; i < args.Length; i++) quoted[i] = Quote(args[i]);
        start.Arguments = string.Join(" ", quoted);
        using (var process = Process.Start(start))
        {
            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();
            File.AppendAllText(log, "  -> real exit " + process.ExitCode + " out=" + stdout + " err=" + stderr + Environment.NewLine);
            Console.Out.Write(stdout);
            Console.Error.Write(stderr);
            return process.ExitCode;
        }
    }
}
