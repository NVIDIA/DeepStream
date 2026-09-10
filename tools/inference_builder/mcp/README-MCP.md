# MCP Server — Agent Reference

Context for MCP-compatible agents using the Inference Builder server.

## Workflow Integration

### Typical Development Workflow

1. **Establish Project Directory**: Before generating any files, confirm a local project directory with the user. All MCP-generated artifacts — pipeline configurations, Dockerfiles, OpenAPI specs, and custom processor modules — should be saved there so they can reference each other and be passed consistently across tool calls.
2. **Explore Samples**: Call `list_resources` and browse `samples://config/*` resources to see available configurations.
3. **Examine Configurations**: Read sample resources to understand configuration patterns.
4. **Create Your Config**: Modify a sample configuration or create your own. When choosing a backend, read `schema://config.schema.json` for the top-level structure, then `schema://index.json` to find the backend-specific parameter schema path, then read that schema for valid parameter options.
5. **Generate Pipeline**: Use `generate_inference_pipeline` to create your code.
6. **Build Container**: Use `build_docker_image` to create a deployable container.
7. **Finalize the Project**: Once smoke tests pass and requirements are verified, save the generated `.tgz` artifact to the local project directory and write a README that summarizes the pipeline, lists all required environment variables and model repository paths, and gives step-by-step instructions for end-user deployment.

### Integration with Version Control

The generated code can be:
- Committed to version control
- Used as a starting point for custom modifications
- Shared with team members
- Deployed to production environments

## Accessing Workspace Files

Tool responses in HTTP mode include a `session_id` field (appended as a trailing content block on every response) and a `url` field on tools that produce files. Use these to fetch artifacts and logs directly from the MCP server over HTTP.

**Remember the MCP server address** (host and port) from the moment you connect — you will need it throughout the session to construct download URLs. Do not rely on the placeholder `{mcp_server}` literally; substitute it with the actual address.

Workspace files are served at:

```
GET http://<mcp_server>/<session_id>/<path>
```

- `GET http://<mcp_server>/<session_id>/` — list all files in the session workspace (JSON)
- `GET http://<mcp_server>/<session_id>/<name>.tgz` — download a generated pipeline archive
- `GET http://<mcp_server>/<session_id>/logs/<container>.log` — fetch a container log
- `GET http://<mcp_server>/<session_id>/logs/` — list all captured logs (JSON)

The `url` field in tool responses already contains the full path template — replace `{mcp_server}` with the address you noted at connection time and `{session_id}` with the value from the trailing `session_id` block.

## Troubleshooting

### Common Issues

**MCP Server Not Found**
- Ensure the MCP server is running
- Check that the path in your MCP client configuration is correct
- Verify Python environment has required dependencies

**Configuration Validation Errors**
- Check that required fields are present in your YAML
- Ensure model definitions are complete
- Verify backend specifications are correct
