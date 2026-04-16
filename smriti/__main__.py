"""Allow running as: python -m smriti"""
try:
    from smriti.mcp_server import main
except ImportError:
    import sys
    print(
        "MCP server requires the 'mcp' extra.\n"
        "Install with: pip install smriti-mem[mcp]\n\n"
        "For library usage:\n"
        "  from smriti import Memory\n"
        "  mem = Memory()",
        file=sys.stderr,
    )
    raise SystemExit(1)

main()
