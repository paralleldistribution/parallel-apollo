from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from artemis.clients.ui_automator_client import (
    UIAutomationConflictError,
    UIAutomatorClient,
)
from artemis.drivers.android.adb_driver import AndroidAdbDriver


@patch("artemis.clients.ui_automator_client.u2.connect")
@patch(
    "artemis.clients.ui_automator_client.ensure_device_awake",
    return_value="host_heartbeat",
)
@patch("artemis.clients.ui_automator_client._ensure_maestro_not_installed")
def test_new_ui_connection_enrolls_device_in_shared_awake_strategy(
    mock_remove_maestro, mock_ensure_awake, mock_connect
):
    client = UIAutomatorClient("device-123")

    client.connect()
    client.connect()

    mock_remove_maestro.assert_called_once_with("device-123")
    mock_ensure_awake.assert_called_once_with("device-123")
    mock_connect.assert_called_once_with("device-123")


@patch("artemis.clients.ui_automator_client.u2.connect", side_effect=RuntimeError("offline"))
@patch(
    "artemis.clients.ui_automator_client.ensure_device_awake",
    return_value="host_heartbeat",
)
@patch("artemis.clients.ui_automator_client._ensure_maestro_not_installed")
def test_failed_ui_connection_does_not_stop_process_awake_service(
    _mock_remove_maestro, mock_ensure_awake, _mock_connect
):
    client = UIAutomatorClient("device-123")

    with pytest.raises(RuntimeError, match="offline"):
        client.connect()

    mock_ensure_awake.assert_called_once_with("device-123")
    assert client._awake_strategy is None


@patch("artemis.clients.ui_automator_client.u2.connect")
@patch(
    "artemis.clients.ui_automator_client.ensure_device_awake",
    return_value="host_heartbeat",
)
@patch("artemis.clients.ui_automator_client._ensure_maestro_not_installed")
def test_client_disconnect_does_not_send_power_cleanup_commands(
    _mock_remove_maestro, mock_ensure_awake, _mock_connect
):
    client = UIAutomatorClient("device-123")
    client.connect()

    client.disconnect()

    mock_ensure_awake.assert_called_once_with("device-123")
    assert client._device is None
    assert client._awake_strategy is None


@pytest.mark.asyncio
async def test_android_driver_disconnect_cleans_up_ui_client():
    ui_client = MagicMock()
    driver = AndroidAdbDriver(
        device_id="device-123",
        adb_client=MagicMock(),
        ui_adb_client=ui_client,
    )

    await driver.disconnect()

    ui_client.disconnect.assert_called_once_with()


@patch("artemis.clients.ui_automator_client.subprocess.run")
@patch("artemis.clients.ui_automator_client.time.sleep")
@patch("artemis.clients.ui_automator_client.ensure_device_awake")
@patch("artemis.clients.ui_automator_client._ensure_maestro_not_installed")
@pytest.mark.parametrize("transient_failure_first", [False, True])
def test_registered_automation_reports_conflict_without_stopping_another_session(
    _maestro, _awake, sleep, run, transient_failure_first
):
    conflict = RuntimeError("UiAutomationService already registered!")
    errors = [RuntimeError("offline"), conflict] if transient_failure_first else [conflict]
    with patch(
        "artemis.clients.ui_automator_client.u2.connect",
        side_effect=errors,
    ) as connect:
        client = UIAutomatorClient("device-123")
        with pytest.raises(UIAutomationConflictError, match="owner must release") as error:
            client.connect()

    assert error.value.__cause__ is conflict
    assert connect.call_count == len(errors)
    assert sleep.call_count == int(transient_failure_first)
    run.assert_not_called()
    assert client._device is None
    assert client._awake_strategy is None


@patch("artemis.clients.ui_automator_client.ensure_device_awake")
@patch("artemis.clients.ui_automator_client._ensure_maestro_not_installed")
def test_disconnect_preserves_server_used_by_another_client(_maestro, _awake):
    first_proxy, second_proxy = MagicMock(), MagicMock()
    second_proxy.dump_hierarchy.return_value = "<hierarchy/>"
    with patch(
        "artemis.clients.ui_automator_client.u2.connect",
        side_effect=[first_proxy, second_proxy],
    ) as connect:
        first, second = UIAutomatorClient("device-123"), UIAutomatorClient("device-123")
        first.connect()
        second.connect()
        first_proxy.reset_mock()
        first.disconnect()
        first.disconnect()

        assert first._device is None
        assert first_proxy.mock_calls == []
        assert second.get_hierarchy() == "<hierarchy/>"
        assert second._device is second_proxy
        assert connect.call_count == 2
        second_proxy.stop_uiautomator.assert_not_called()


@patch("artemis.clients.ui_automator_client.time.sleep")
@patch("artemis.clients.ui_automator_client.ensure_device_awake")
@patch("artemis.clients.ui_automator_client._ensure_maestro_not_installed")
def test_dead_proxy_reconnect_does_not_wait_for_server_teardown(_maestro, _awake, sleep):
    dead_proxy, recovered_proxy = MagicMock(), MagicMock()
    type(dead_proxy).info = PropertyMock(side_effect=RuntimeError("broken session"))
    recovered_proxy.dump_hierarchy.return_value = "<hierarchy/>"
    with patch(
        "artemis.clients.ui_automator_client.u2.connect",
        side_effect=[dead_proxy, recovered_proxy],
    ) as connect:
        client = UIAutomatorClient("device-123")
        client.connect()
        dead_proxy.reset_mock()

        assert client.get_hierarchy() == "<hierarchy/>"

    assert client._device is recovered_proxy
    assert connect.call_count == 2
    assert dead_proxy.mock_calls == []
    sleep.assert_not_called()
